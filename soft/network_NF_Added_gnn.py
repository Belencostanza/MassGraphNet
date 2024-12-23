import os
import torch
import math

from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import MetaLayer
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torchvision import transforms
from torch_geometric.loader import DataLoader

#from NFScripts.archi import createAchi, create_condDist
from archi import createAchi, create_condDist

import optuna

from load_data_nopool import training_set, validation_set, test_set#, study_name


min_valid_loss = 1000                       #set this to a large number. Used to compute

batch_size     = 256                        #number of elements each batch contains. Hyper-parameter
dr             = 0.0                       #dropout rate. Hyper-parameter
epochs         = 100                       #number of epochs to train the network. Hyper-parameter


f_model = "models_all_more4_globalnorm_meanall_newrange/"
f_losses = "losses_all_more4_globalnorm_meanall_newrange/"
study_name = "NF2.0_more4_globalnorm_meanall_newrange"


torch.manual_seed(12345)


if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


#Neural network based on CosmoGraphNet to predict the WDM mass of the graphs 

#MPL for update edges attributes
class EdgeModel(nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=True, norm=False):
        super().__init__()
        
        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in, hidden_channels),
                  ReLU(),
                  Linear(hidden_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
    
        return out

#MPL for update nodes attributes
class NodeModel(nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=True, norm=False):
        super().__init__()
        
        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + 3*edge_out + global_in, hidden_channels),
                  ReLU(),
                  Linear(hidden_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))


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
        if self.residuals:
            out = out + x
    
        return out


class GNN(nn.Module):
    def __init__(self, u_dim, node_features, n_layers, hidden_dim, dim_out, residuals=True):
        super().__init__()

        self.n_layers = n_layers
        self.dim_out = dim_out


        node_in = node_features
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        edge_out = hidden_dim
        
        global_in = u_dim
        node_out = hidden_dim 
        hidden_channels = hidden_dim
        
        #encoder graph block
        layers = []

        inlayer = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=False),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=False))

        layers.append(inlayer)

        node_in = node_out
        edge_in = edge_out

        #hidden graph block
        #layers = []
        for i in range(n_layers-1):
            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=residuals),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hidden_channels, global_in, residuals=residuals))
            layers.append(lay)

        self.layers = ModuleList(layers)

        self.outlayer = Sequential(Linear(3*node_out + global_in, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, self.dim_out))

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        
        for layer in self.layers:
            #print('hello')
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u, batch)

        # Multipooling layer
        addpool = global_add_pool(x, batch)
        meanpool = global_mean_pool(x, batch)
        maxpool = global_max_pool(x, batch)

        out = torch.cat([addpool,meanpool,maxpool, u], dim=1)
        # Final linear layer
        out = self.outlayer(out)

        return out
        
        
        
def train(dist_x2_given_x1, model,optimizer,criterion,scheduler, train_loader = None):
  train_loss = 0.0
  model.train()
  
  for data in train_loader:# Iterate in batches over the training dataset.
    data.to(device)
    optimizer.zero_grad()
    # Previous GNN part ############################
    out = model(data)
    y_target = data.y 
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    
    # loss_mse = criterion(out, y_target)
    # loss = torch.log(loss_mse) #probamos poniendo log
    # #################################################
    # print(f"out: {out}", flush = True)
    # print(f"target: {y_target}", flush = True)
    out = torch.nan_to_num(out)
    ln_p_x2_given_x1 = dist_x2_given_x1.condition(out).log_prob(y_target)
    loss = -(ln_p_x2_given_x1).mean()
    
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    scheduler.step()
    train_loss += loss.item()
  last_loss = train_loss/len(train_loader)
  
  return last_loss

def eval(dist_x2_given_x1, model,criterion, trial, modules, best_loss, valid_loader, hyperparameters = []):
  valid_loss = 0.0
  model.eval()
  predicted = np.array([])
  true = np.array([])
  
  for data in valid_loader:
    with torch.no_grad():
        data.to(device)
        out = model(data)
        y_target = data.y
        y_target = torch.reshape(y_target, (data.num_graphs, 1))
    
        # loss_mse = criterion(out, y_target)
        # loss = torch.log(loss_mse)
        out = torch.nan_to_num(out)
        ln_p_x2_given_x1 = dist_x2_given_x1.condition(out).log_prob(y_target)
        loss = -(ln_p_x2_given_x1).mean()
    
        valid_loss += loss.item()

    # if save:
    #     predicted = np.append(predicted, out.detach().cpu().numpy())
    #     true = np.append(true, y_target.detach().cpu().numpy())
  
  val_loss = valid_loss/len(valid_loader)

  if val_loss < best_loss:
      print('saving')
      fmodel = f_model
      torch.save(model.state_dict(), fmodel + name_model(hyperparameters))
      torch.save(dist_x2_given_x1, fmodel + f"dist_x2_given_x1_{trial.number}_{study_name}.pt")
      torch.save(modules, fmodel + f"modules_{trial.number}_{study_name}.pt")
      torch.save(modules.state_dict(), fmodel + f"modules_state_dict_{trial.number}_{study_name}.pt")
      best_loss = val_loss


  return val_loss, best_loss

def normalize_node_features(train_dataset, valid_dataset, test_dataset, mean_node = None, std_node = None):
    Combined_Dataset = train_dataset + valid_dataset + test_dataset
    if mean_node is None:
        mean_node = torch.mean(torch.cat([data.x for data in Combined_Dataset], dim=0), dim=0)
        std_node = torch.std(torch.cat([data.x for data in Combined_Dataset], dim=0), dim=0)
        mean_u = torch.mean(torch.cat([data.u for data in Combined_Dataset], dim=0), dim=0)
        std_u = torch.std(torch.cat([data.u for data in Combined_Dataset], dim=0), dim=0)

    print('global mean:', mean_node)
    print('global std:', std_node)
    print('global u:', mean_u)
    print('std u:', std_u) 
    for data in train_dataset:
        data.x = (data.x - mean_node) / std_node
        data.u = (data.u - mean_u) / std_u
    for data in valid_dataset:
        data.x = (data.x - mean_node) / std_node
        data.u = (data.u - mean_u) / std_u
    for data in test_dataset:
        data.x = (data.x - mean_node) / std_node
        data.u = (data.u - mean_u) / std_u

    return train_dataset, valid_dataset, test_dataset
    
def name_model(hyperparameters):
    return "n_layers_" + str(hyperparameters[0]) +  "_n_units_" + str(hyperparameters[1]) + "_lr_" + "{:.3e}".format(hyperparameters[2]) + "_wd_" + "{:.3e}".format(hyperparameters[3], 3) + "_rlink_" + "{:.3e}".format(hyperparameters[4], 3)
        
#define an objective function to be minimized by the loss function
def objective(trial):
    # Create the dataset based on the link 
    #r_link = trial.suggest_float('r_link', 1e-3, 2e-1)
    #1e-5 a 1e-3
    r_link = trial.suggest_float("r_link", 1.e-8, 1.e-4, log=True) #cambie esto que estaba de -4 a -2
    train_dataset = training_set(r_link)
    valid_dataset = validation_set(r_link)
    test_dataset = test_set(r_link)

    train_dataset, valid_dataset, test_dataset = normalize_node_features(train_dataset, valid_dataset, test_dataset, mean_node = None, std_node = None)

    u = train_dataset[0].u
    u_dim = u.shape[1]

    #print('finish')
    
    
    #suggest values in the number of layers
    n_layers = trial.suggest_int('n_layers',1,5)
    #suggest values in the number of neurons
    #n_units = trial.suggest_int('n_units',64,128)
    n_units = trial.suggest_categorical("n_units", [64, 128, 256, 512])

    NFHyper = suggest_NF_Hyper_Parameters(trial)
    
    latent_dim = 64  #before was in 8
    model = GNN(u_dim = u_dim, node_features = 14, n_layers = n_layers, hidden_dim = n_units, dim_out = latent_dim, residuals=True)
        
    
    criterion = nn.MSELoss()  #loss function. In this case MSE (mean squared error)
    
    lr = trial.suggest_float('lr', 1e-9,1e-5, log=True)
    #lr = trial.suggest_float('lr', 1e-5,1e-1)
    wd = trial.suggest_float('wd', 1e-9,1e-6, log=True)
    
    # NF models
    modules, dist_x2_given_x1 = createNFAchi(NFHyper, lr, wd, 1, latent_dim);

    # combine two models parameter
    combined_parameters = list(model.parameters()) + list(modules.parameters())
    
    optimizer = torch.optim.Adam(combined_parameters, lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1.e-3, cycle_momentum=False, step_size_up=500)

    model.to(device=device)   

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=n_CPU)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)#, num_workers=n_CPU)

    best_loss = 1e7
    print('starting with:',best_loss)

    trainLoss_history = []
    validLoss_history = []

    for epoch in range(epochs):
        print(epoch, flush= True) #me printea las epocas en cada run
        train_loss = train(dist_x2_given_x1 ,model,optimizer,criterion, scheduler, train_loader)
        valid_loss, best_loss = eval( dist_x2_given_x1, model,criterion, trial, modules, best_loss, valid_loader, hyperparameters = [n_layers, n_units, lr, wd, r_link])

        print(f"train loss {train_loss}, valid loss {valid_loss}", flush = True)
        
        if(math.isnan(train_loss)):
            return 10000
        
        trainLoss_history.append(train_loss)
        validLoss_history.append(valid_loss)

    # Save the output data in a file
    # This can be changed to test_loader when doing the final test  

    #eval(dist_x2_given_x1 ,model,criterion, valid_loader, save = True, hyperparameters = [n_layers, n_units, lr, wd, r_link])

    #plt.clf()
    #plt.plot(trainLoss_history, label='train_loss')
    #plt.plot(validLoss_history,label='val_loss')
    #plt.legend()
    #plt.show()
    #plt.savefig("losses/" + name_model([n_layers, n_units, lr, wd]) + ".png")
    #plt.close()
    np.savez(f_losses + name_model([n_layers, n_units, lr, wd, r_link]), trainLoss_history, validLoss_history)

    return best_loss


def suggest_NF_Hyper_Parameters(trial):
    condition_layers = trial.suggest_int("condition_layers", 1,15)
    count_bins = trial.suggest_int("NFcount_bins", 4, 256)
    

    dim_options = trial.suggest_categorical("dim_options", [64, 128, 256])
    hidden_dims = [dim_options,dim_options]
    
    
    return [condition_layers, count_bins, hidden_dims]

#def suggest_NF_Hyper_Parameters(trial):
#    condition_layers = trial.suggest_int("condition_layers", 1,15)
#    count_bins = trial.suggest_int("NFcount_bins", 4, 256)
    
    # TODO:To choose from a list
#    hidden = trial.suggest_int("NFhidden_dims", 0, 4)

#    dim_options = [16,32,64,128,256]
#    hidden_dims = [dim_options[hidden],dim_options[hidden]]
    
    
#    return [condition_layers, count_bins, hidden_dims]

def createNFAchi( hyperparameters, lr, wd,target_dimention, context_dimension):
    condition_layers = hyperparameters[0]
    count_bins = hyperparameters[1]
    hidden_dims = hyperparameters[2]
    
    dist_x2_given_x1, x2_transform = create_condDist(target_dimention,context_dimension,condition_layers, count_bins,hidden_dims ,device)
    
    modules, optimizer = createAchi(x2_transform, lr, wd, device)
    # The optimizer will not be used here
    
    return modules, dist_x2_given_x1
    
if __name__ == '__main__':
        
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage, load_if_exists= True)
    study.optimize(objective, n_trials=25)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))


    print("Best trial:")
    trial = study.best_trial

    print("value:", trial.value)

    print("Params:")
    for key, value in trial.params.items():
        print("{}:{}".format(key, value))
        
