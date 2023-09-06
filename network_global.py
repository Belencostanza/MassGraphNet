import os
import torch

#%matplotlib inline
#import networkx as nx
#import matplotlib.pyplot as plt

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


min_valid_loss = 1e7                       #set this to a large number. Used to compute

batch_size     = 8                        #number of elements each batch contains. Hyper-parameter
lr             = 1e-5                      #value of the learning rate. Hyper-parameter
wd             = 0.0                       #value of the weight decay. Hyper-parameter
dr             = 0.0                       #dropout rate. Hyper-parameter
epochs         = 300                       #number of epochs to train the network. Hyper-parameter

#name of the model
f_model = 'model_omega_300.pt'

#name of the loss 
name_loss = 'omega_loss_300'

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
train_dataset = torch.load('omega_train.pt')
valid_dataset = torch.load('omega_valid.pt')

print('finish')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


#for step, data in enumerate(train_loader):
#    print(f'Step {step + 1}:')
#    print('=======')
#    print(f'Number of graphs in the current batch: {data.num_graphs}')
#    print(data)
#    print()



#Neural network based on CosmoGraphNet to predict the WDM mass of the graphs 

#MPL for update edges attributes
class EdgeModel(nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_dim):
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
    def __init__(self, node_in, node_out, edge_in, edge_out, hidden_dim):
        super().__init__()

        layers = [Linear(node_in + 3*edge_in + 1, hidden_dim),
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
    def __init__(self, node_features, n_layers, hidden_dim, dim_out):
        super().__init__()

        self.n_layers = n_layers
        self.dim_out = dim_out
        # Number of input node features = 10
        node_in = node_features
        edge_in = 1
        edge_out = 1
        node_out = node_features 
        hidden_dim = hidden_dim

        #node_in = node_out
        #edge_in = edge_out

        layers = []
        for i in range(n_layers-1):
            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hidden_dim),
                        edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hidden_dim))
            layers.append(lay)

        self.layers = ModuleList(layers)

        self.outlayer = Sequential(Linear(3*node_out + 1, hidden_dim),
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
    
#las node features son 10
model = GNN(node_features = 12, n_layers = 3, hidden_dim = 64, dim_out = 1)
criterion = nn.MSELoss()  #loss function. In this case MSE (mean squared error)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)


model.to(device=device)

def train():
  train_loss = 0.0
  model.train()
  for data in train_loader:# Iterate in batches over the training dataset.
    data.to(device)
    optimizer.zero_grad()
    out = model(data)
    y_target = data.y 
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    loss = criterion(out, y_target)

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    train_loss += loss.item()
  last_loss = train_loss/len(train_loader)
  #print('Training loss =', last_loss)
  return last_loss

def eval():
  valid_loss = 0.0
  model.eval()
  for data in valid_loader:
    data.to(device)
    out = model(data)
    y_target = data.y
    y_target = torch.reshape(y_target, (data.num_graphs, 1))
    loss = criterion(out, y_target)
    valid_loss += loss.item()
  val_loss = valid_loss/len(valid_loader)
  #print('test loss =', val_loss)
  return val_loss

train_epoch=[]
valid_epoch=[]
for epoch in range(epochs):
  train_loss = train()
  valid_loss = eval()
  train_epoch.append(train_loss)
  valid_epoch.append(valid_loss)
  print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test loss: {valid_loss:.4f}')

  if valid_loss<min_valid_loss:
      torch.save(model.state_dict(), f_model)
      min_valid_loss = valid_loss
      print(' (best-model)')

np.savez(name_loss, train_epoch, valid_epoch)
