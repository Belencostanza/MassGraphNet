import os
import torch


from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import MetaLayer
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torchvision import transforms
from torch_geometric.loader import DataLoader

import optuna

from load_data_mass import training_set, validation_set, test_set, n_CPU

from accelerate import Accelerator
accelerator = Accelerator()


min_valid_loss = 1e7                       #set this to a large number. Used to compute

batch_size     = 8                        #number of elements each batch contains. Hyper-parameter

#lr             = 1e-5                      #value of the learning rate. Hyper-parameter
#wd             = 0.0                       #value of the weight decay. Hyper-parameter
dr             = 0.0                       #dropout rate. Hyper-parameter
epochs         = 1                       #number of epochs to train the network. Hyper-parameter

#name of the model
#f_model = 'model_mass_300.pt'

#name of the loss 
#name_loss = 'mass_loss_300'

torch.manual_seed(12345)
#data_list = data_list.shuffle() #no le hice un shuffle inicial en la data

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


print('reading data')
#load the dataset
# train_dataset = torch.load('masswdm_train_menos10_all.pt')
# valid_dataset = torch.load('masswdm_valid_menos10_all.pt')


#for step, data in enumerate(train_loader):
#    print(f'Step {step + 1}:')
#    print('=======')
#    print(f'Number of graphs in the current batch: {data.num_graphs}')
#    print(data)
#    print()



#Neural network based on CosmoGraphNet to predict the WDM mass of the graphs 

#MPL for update edges attributes
class EdgeModel(nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_dim, global_in):
        super().__init__()

        layers = [Linear(node_in*2 + edge_in, hidden_dim),
                  ReLU(),
                  Linear(hidden_dim, edge_out)]

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        return out

#MPL for update nodes attributes
class NodeModel(nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_dim, global_in):
        super().__init__()

        layers = [Linear(node_in + 3*edge_in + global_in, hidden_dim),
                  ReLU(),
                  Linear(hidden_dim, node_out)]

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)
        return out


class GNN(nn.Module):
    def __init__(self, u_dim, node_features, n_layers, hidden_dim, dim_out):
        super().__init__()

        self.n_layers = n_layers
        self.dim_out = dim_out
        # Number of input node features = 12
        node_in = node_features
        edge_in = 3
        edge_out = 3
        global_in = u_dim
        node_out = node_features 
        hidden_dim = hidden_dim

        #node_in = node_out
        #edge_in = edge_out

        layers = []
        for i in range(n_layers-1):
            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hidden_dim, global_in),
                        edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hidden_dim, global_in))
            layers.append(lay)

        self.layers = ModuleList(layers)

        self.outlayer = Sequential(Linear(3*node_out + global_in, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, self.dim_out))

    def forward(self, data):
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        
        for layer in self.layers:
            #print('hello')
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u, batch=data.batch)

        # Multipooling layer
        addpool = global_add_pool(x, data.batch)
        meanpool = global_mean_pool(x, data.batch)
        maxpool = global_max_pool(x, data.batch)

        out = torch.cat([addpool,meanpool,maxpool, u], dim=1)
        # Final linear layer
        out = self.outlayer(out)

        return out

def train(model,optimizer,criterion, train_loader = None):
  train_loss = 0.0
  model.train()
  
  model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
  
  for data in train_loader:# Iterate in batches over the training dataset.
    data.to(device)
    optimizer.zero_grad()
    out = model(data)
    y_target = data.y 
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    loss = criterion(out, y_target)

    # loss.backward()  # Derive gradients.
    accelerator.backward(loss)
    optimizer.step()  # Update parameters based on gradients.
    train_loss += loss.item()
  last_loss = train_loss/len(train_loader)
  #print('Training loss =', last_loss)
  return last_loss

def eval(model,criterion, valid_loader = None, save = False, hyperparameters = []):
  valid_loss = 0.0
  model.eval()
  predicted = np.array([])
  true = np.array([])
  for data in valid_loader:
    data.to(device)
    out = model(data)
    y_target = data.y
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    loss = criterion(out, y_target)
    valid_loss += loss.item()

    if save:
        predicted = np.append(predicted, out.detach().cpu().numpy())
        true = np.append(true, y_target.detach().cpu().numpy())
  
  val_loss = valid_loss/len(valid_loader)
  #print('test loss =', val_loss)
  if save:
    with open('outputs/' + name_model(hyperparameters) + ".npy", 'wb') as f:
        np.save(f, predicted)
        np.save(f, true)
    
    torch.save(model.state_dict(), "models/" + name_model(hyperparameters))
        
  return val_loss
    
def name_model(hyperparameters):
    return "n_layers_" + str(hyperparameters[0]) +  "_n_units_" + str(hyperparameters[1]) + "_lr_" + "{:.3e}".format(hyperparameters[2]) + "_wd_" + "{:.3e}".format(hyperparameters[3], 3) + "_rlink_" + "{:.3e}".format(hyperparameters[4], 3)
        
#define an objective function to be minimized by the loss function
def objective(trial):
    # Create the dataset based on the link 
    r_link = trial.suggest_float('r_link', 1e-3, 2e-1)
    train_dataset = training_set(r_link)
    valid_dataset = validation_set(r_link)

    u = train_dataset[0].u
    u_dim = u.shape[1]

    print('finish')
    
    
    #suggest values in the number of layers
    n_layers = trial.suggest_int('n_layers',2,3)
    #suggest values in the number of neurons
    n_units = trial.suggest_int('n_units',64,128)

    model = GNN(u_dim = u_dim, node_features = 14, n_layers = n_layers, hidden_dim = n_units, dim_out = 1)

    criterion = nn.MSELoss()  #loss function. In this case MSE (mean squared error)

    
    lr = trial.suggest_float('lr', 1e-8,1e-3)
    wd = trial.suggest_float('wd', 1e-9,1e-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)

    model.to(device=device)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)

    for epoch in range(epochs):
        print(epoch, flush= True)
        train_loss = train(model,optimizer,criterion, train_loader)
        valid_loss = eval(model,criterion, valid_loader)

    # Save the output data in a file
    # This can be changed to test_loader when doing the final test
    eval(model,criterion, valid_loader, save = True, hyperparameters = [n_layers, n_units, lr, wd, r_link])

    return valid_loss

storage = "sqlite:///example_all.db"
study = optuna.create_study(study_name="bTest", direction='minimize', storage=storage, load_if_exists= True)
study.optimize(objective, n_trials=1)

#pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
#complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
#print("  Number of pruned trials: ", len(pruned_trials))
#print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("value:", trial.value)

print("Params:")
for key, value in trial.params.items():
    print("{}:{}".format(key, value))
    
