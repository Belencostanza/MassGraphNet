import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from load_data_symmetries import training_set, validation_set, test_set, n_CPU
from network_optuna_symmetries import GNN, batch_size
import seaborn as sns

from tarp import get_drp_coverage

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
    
study_name = "NF1.0"
storage = "sqlite:///NF1.0.db"

study = optuna.load_study(study_name=study_name, storage=storage)


trial = study.best_trial

# Get All the hyper parameters
n_layers = trial.params["n_layers"]
n_units = trial.params["n_units"]
lr = trial.params["lr"]
wd = trial.params["wd"]
r_link = trial.params["r_link"]


print("\nTrial number {}".format(trial.number), flush=True)
print("Value: %.5e"%trial.value, flush=True)
print(" Params: ", flush=True)
for key, value in trial.params.items():
    print("    {}: {}".format(key, value), flush=True)

def name_model(hyperparameters):
    return "n_layers_" + str(hyperparameters[0]) +  "_n_units_" + str(hyperparameters[1]) + "_lr_" + "{:.3e}".format(hyperparameters[2]) + "_wd_" + "{:.3e}".format(hyperparameters[3], 3) + "_rlink_" + "{:.3e}".format(hyperparameters[4], 3)

resultsname = name_model([n_layers, n_units, lr, wd, r_link])

# train_dataset = training_set(r_link)
valid_dataset = validation_set(r_link)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)

u = valid_dataset[0].u
u_dim = u.shape[1]

latent_dim = 8

# Load the GNN Model
model = GNN(u_dim = u_dim, node_features = 15, n_layers = n_layers, hidden_dim = n_units, dim_out = latent_dim, residuals=True)  

model.load_state_dict(torch.load("models/" + resultsname))

dist_x2_given_x1 = torch.load(f"models/dist_x2_given_x1_{trial.number}.pt")
modules = torch.load(f"models/modules_{trial.number}.pt")
modules.load_state_dict(torch.load(f"models/modules_state_dict_{trial.number}.pt"))

oneSamplePosterior = []

def train(dist_x2_given_x1, model,optimizer,scheduler, train_loader = None):
  model.to(device)
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

def eval(dist_x2_given_x1, model, valid_loader = None):
  model.to(device)
  model.eval()
  predicted = np.array([])
  true = np.array([])
  error = np.array([])
  flowSamples = np.array([])
  
  flowsampleNumber = 0
  
  i = 0
  for data in valid_loader:
    with torch.no_grad():
        data.to(device)
        out = model(data)
        y_target = data.y
        y_target = torch.reshape(y_target, (data.num_graphs, 1))
    
        out = torch.nan_to_num(out)
        flowSample = dist_x2_given_x1.condition(out).sample(torch.Size([1000,out.shape[0]]))
        
        flowMean = torch.mean(flowSample, dim=0)
        flowSTD = torch.std(flowSample, dim=0)
        
        true = np.append(true, y_target.detach().cpu().numpy())
        
        predicted = np.append(predicted, flowMean.detach().cpu().numpy())
        
        error = np.append(error, flowSTD.detach().cpu().numpy())
        
        if i == 0:
            flowSamples = flowSample.squeeze().detach().cpu().numpy().transpose()
            print(flowSamples.shape, flush=True)
            print(flowSamples, flush=True)
        else:
            print(flowSamples.shape, flush=True)
            print(flowSample.squeeze().detach().cpu().numpy().transpose().shape, flush=True)
            flowSamples = np.vstack((flowSamples, flowSample.squeeze().detach().cpu().numpy().transpose()), )
            
        flowsampleNumber += flowSample.shape[1]
        print(flowsampleNumber, flush=True)
        
        if i == 100:
            flowSamples = flowSample.squeeze().detach().cpu().numpy().transpose()[5]
        # else:
        #     flowSamples = np.append(flowSamples, flowSample.squeeze().detach().cpu().numpy().transpose(), )
                
        i+=1
  
  print(flowSamples.shape, flush=True)
  plt.clf()
  
  
  print(true, flush=True)
  print(predicted, flush=True)
  plt.scatter(true, predicted, s=1)
#   plt.errorbar(true, predicted, yerr=error, fmt='o', markersize=1, lw = 0.5)
  xref = np.linspace(0.03,0.58,100)
  yref = xref
  plt.plot(xref, yref, color='red', linestyle='-')
  plt.savefig("results/" + "NFTest5" + ".png")

  # TODO: Add test loss
  plt.clf()
  print(flowSamples.shape, flush=True)
  sns.distplot(flowSamples, hist=False, kde=True,
        bins=None, color='firebrick',
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2},
        label='flow')
  print(true[31], flush=True)
  plt.grid()
  plt.xlim(-0.1,0.6)

  plt.savefig("results/" + "NFTest1OnePosterior5" + ".png")
  
  plt.clf()
  flowSamples = flowSamples.transpose()[:,:,np.newaxis]
  print(flowSamples.shape, flush=True)
  print(true.shape, flush=True)
  coverage = get_drp_coverage(flowSamples, true[:,np.newaxis])
  plt.plot(np.array(coverage)[0],np.array(coverage)[1])
  plt.savefig("results/coverage.png")
  
  return 0

# combined_parameters = list(model.parameters()) + list(modules.parameters())
# optimizer = torch.optim.Adam(combined_parameters, lr=lr, betas=(0.5, 0.999), weight_decay=wd)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1.e-3, cycle_momentum=False, step_size_up=500)

# for i in range(77):
#     trainloss = train(dist_x2_given_x1, model, optimizer, scheduler, train_loader)
    
#     print(f"Epoch {i} - Train Loss: {trainloss}", flush=True)
eval(dist_x2_given_x1, model, valid_loader)






