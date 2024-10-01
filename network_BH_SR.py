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

from NFScripts.archi import createAchi, create_condDist

from torch.utils.tensorboard import SummaryWriter

import optuna
import sys

from load_data_BH import training_set, validation_set, test_set, n_CPU, device, study_name




batch_size     = 256                        #number of elements each batch contains. Hyper-parameter
dr             = 0.0                       #dropout rate. Hyper-parameter
epochs         = 300                       #number of epochs to train the network. Hyper-parameter



torch.manual_seed(12345)
writer = SummaryWriter()


#Neural network based on CosmoGraphNet to predict the WDM mass of the graphs 


#MPL for update nodes attributes
class NodeModel(nn.Module):
    def __init__(self, node_in, node_out, hidden_channels, global_in, residuals=True, norm=False):
        super().__init__()
        
        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + global_in, hidden_channels),
                  ReLU(),
                  Linear(hidden_channels, hidden_channels),
                  ReLU(),
                  Linear(hidden_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))


        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        
        out = torch.cat([x, u[batch]], dim=1)

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
        
        global_in = u_dim
        node_out = 1
        hidden_channels = hidden_dim
        
        #encoder graph block
        layers = []

        inlayer = MetaLayer(node_model=NodeModel(node_in, node_out, hidden_channels, global_in, residuals=False))
        

        layers.append(inlayer)

        node_in = node_out

        #hidden graph block
        #layers = []
        for i in range(n_layers-1):
            lay = MetaLayer(node_model=NodeModel(node_in, node_out, hidden_channels, global_in, residuals=residuals))
                            
            layers.append(lay)

        self.layers = ModuleList(layers)

        # self.outlayer = Sequential(Linear(node_out + global_in, hidden_channels),
        self.outlayer = Sequential(Linear(node_out, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, self.dim_out))

    def create_mask(self, x):
        # Selección aleatoria de un subconjunto de nodos
        # subset_random = np.rand(1, requires_grad=False)*0.5 + 0.5
        subset_size = int(90*x.size(0)/100)
        # subset_size = int(subset_random*x.size(0))
        mask = torch.zeros_like(x)
        indices = torch.randperm(x.size(0))[:subset_size]
        mask[indices] = 1
        return mask
    
    
    
    def forward(self, data, use_mask = False):
        x, u, batch = data.x, data.u, data.batch
        
        # input_value = torch.hstack((x,u[batch]))
        
        for layer in self.layers:
        #     print('hello')
            edge_index = np.zeros((2,1))
            edge_attr = np.zeros((1,1))
            x, edge_attr, _ = layer(x, edge_index, edge_attr, u, batch)
            
        
        addpool = global_add_pool(x, batch)

        out = torch.cat([addpool], dim=1)
        # Final linear layer
        out = self.outlayer(out)
        
      
        # add_output = torch.hstack((addpool , out))
        # f = open('./SRdata/add_output.dat', 'a')
        # np.savetxt(f, add_output.detach().cpu().numpy())
        # f.close()
        
        return out
        
        
        
def train(model,optimizer,criterion,scheduler, train_loader = None):
  train_loss = 0.0
  model.train()
  
  for data in train_loader:# Iterate in batches over the training dataset.
    data.to(device)
    optimizer.zero_grad()
    # Previous GNN part ############################
    out = model(data)
    y_target = data.y 
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    
    loss = criterion(out, y_target)
    
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    scheduler.step()
    # print(scheduler.get_last_lr(), flush=True)
    train_loss += loss.item()
  last_loss = train_loss/len(train_loader)
  
  return last_loss

def eval(model,criterion, valid_loader = None, save = False, hyperparameters = []):
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
    
        
        # out = torch.nan_to_num(out)
        # ln_p_x2_given_x1 = dist_x2_given_x1.condition(out).log_prob(y_target)
        
        loss = criterion(out, y_target)
    
        valid_loss += loss.item()
  
  val_loss = valid_loss/len(valid_loader)

  if save:
    fmodel = "models/"
    torch.save(model.state_dict(), "models/" + name_model(hyperparameters))
    
    
  return val_loss
    
def name_model(hyperparameters):
    return "n_layers_" + str(hyperparameters[0]) +  "_n_units_" + str(hyperparameters[1]) + "_lr_" + "{:.3e}".format(hyperparameters[2]) + "_wd_" + "{:.3e}".format(hyperparameters[3], 3) + "_rlink_" + "{:.3e}".format(hyperparameters[4], 3)


def normalize_node_features(train_dataset, valid_dataset, test_dataset, mean_node = None, std_node = None):

    Combined_Dataset = train_dataset + valid_dataset + test_dataset

    if mean_node is None:
        mean_node = torch.mean(torch.cat([data.x for data in Combined_Dataset], dim=0), dim=0)
        std_node = torch.std(torch.cat([data.x for data in Combined_Dataset], dim=0), dim=0)

    print(mean_node, std_node, flush = True)
    for data in train_dataset:
        data.x = (data.x - mean_node) / std_node
    for data in valid_dataset:
        data.x = (data.x - mean_node) / std_node
    for data in test_dataset:
        data.x = (data.x - mean_node) / std_node
    return train_dataset, valid_dataset, test_dataset

def saveModel(model, hyperparameters):
    torch.save(model.state_dict(), "models/" + name_model(hyperparameters))

#define an objective function to be minimized by the loss function
def objective(trial):
    # declear min_valid_loss as global variable
    min_valid_loss = 1000
    
    # Create the dataset based on the link 
    r_link = trial.suggest_float("r_link", 1.e-4, 1.e-2, log=True)
    train_dataset = training_set(r_link)
    valid_dataset = validation_set(r_link)
    test_dataset = test_set(r_link)

    train_dataset, valid_dataset, test_dataset = normalize_node_features(train_dataset, valid_dataset, test_dataset)
    
    
    # valid_dataset, _, _ = normalize_node_features(valid_dataset, mean_node, std_node)

    u = train_dataset[0].u
    u_dim = u.shape[1]

    #suggest values in the number of layers
    n_layers = trial.suggest_int('n_layers',1,5)
    #suggest values in the number of neurons
    n_units = trial.suggest_categorical("n_units", [64, 128, 256, 512])

    
    latent_dim = 1
    model = GNN(u_dim = u_dim, node_features = 1, n_layers = n_layers, hidden_dim = n_units, dim_out = latent_dim, residuals=True)
    
    
    # This will no longer be used in the NF model
    
    lr = trial.suggest_float('lr', 1e-10,1e-2, log=True)
    #lr = trial.suggest_float('lr', 1e-5,1e-1)
    wd = trial.suggest_float('wd', 1e-9,1e-6, log=True)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1.e-4, cycle_momentum=False, step_size_up=1000)
    criterion = torch.nn.MSELoss()
    
    model.to(device=device)   

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)

    # trainLoss_history = []
    # validLoss_history = []

    hyperparameters = [n_layers, n_units, lr, wd, r_link]

    for epoch in range(epochs):
        print(epoch, flush= True) #me printea las epocas en cada run
        train_loss = train(model,optimizer,criterion, scheduler, train_loader)
        valid_loss = eval(model,criterion, valid_loader)


        print(f"train loss {train_loss}, valid loss {valid_loss}", flush = True)
        writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        
        if(math.isnan(train_loss)):
            return 10000
        
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            saveModel(model, hyperparameters)
            print("Model saved", flush = True)
        # trainLoss_history.append(train_loss)
        # validLoss_history.append(valid_loss)

    writer.flush()
    # Save the output data in a file
    # This can be changed to test_loader when doing the final test  

    eval(model,criterion, valid_loader, save = False, hyperparameters=hyperparameters)
   
    fmodel = "models/"
    
    # np.savez("losses/" + name_model([n_layers, n_units, lr, wd, r_link]), trainLoss_history, validLoss_history)

    return min_valid_loss
    
if __name__ == '__main__':
        
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage, load_if_exists= True)
    study.optimize(objective, n_trials=50)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))


    print("Best trial:")
    trial = study.best_trial

    print("value:", trial.value)

    print("Params:")
    for key, value in trial.params.items():
        print("{}:{}".format(key, value))
        
