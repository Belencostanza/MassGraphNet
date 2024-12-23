import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from load_data_nopool import training_set, validation_set, test_set#, study_name
from network_NF_Added_deep import GNN, batch_size, name_model, study_name, f_model
import seaborn as sns
import math

from tarp import get_drp_coverage

import pandas as pd

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14

plt.rcParams['mathtext.rm'] = 'DeJavu Serif'
plt.rcParams['mathtext.it'] = 'DeJavu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DeJavu Serif:bold'

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda:0')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

f_result = "results_all_more4_globalnorm_meanmtot/"
    
#study_name = "NF2.0DeepSetMBH"
storage = f"sqlite:///{study_name}.db"

study = optuna.load_study(study_name=study_name, storage=storage)

#trial_number = 0
trial = study.best_trial
#trials = study.trials
#trial = trials[trial_number]

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


resultsname = name_model([n_layers, n_units, lr, wd, r_link])


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

train_dataset = training_set(r_link)
valid_dataset = validation_set(r_link)
test_dataset = test_set(r_link)

train_dataset, valid_dataset, test_dataset = normalize_node_features(train_dataset, valid_dataset, test_dataset, mean_node = None, std_node = None)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_CPU)
valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


u = valid_dataset[0].u
u_dim = u.shape[1]

# latent_dim = 64 for NF2.0
latent_dim = 64

# Load the GNN Model
model = GNN(u_dim = u_dim, node_features = 14, n_layers = n_layers, hidden_dim = n_units, dim_out = latent_dim, residuals=True)  
# model = GNN(u_dim = u_dim, node_features = 9, n_layers = n_layers, hidden_dim = n_units, dim_out = latent_dim, residuals=True)  

model.load_state_dict(torch.load(f_model + resultsname, map_location=device))

dist_x2_given_x1 = torch.load(f_model + f"dist_x2_given_x1_{trial.number}_{study_name}.pt", map_location=device)
modules = torch.load(f_model + f"modules_{trial.number}_{study_name}.pt", map_location=device)
modules.load_state_dict(torch.load(f_model + f"/modules_state_dict_{trial.number}_{study_name}.pt", map_location=device))

oneSamplePosterior = []


def calculate_metrics(true, predict, error):
    chi_squared  = np.mean(np.power(predict - true, 2)/ np.power(error, 2))

    RSS = np.square(np.subtract(true, predict)).sum()
    TSS = np.square(np.subtract(true, np.mean(true))).sum()

    R_squared = 1 - RSS/TSS

    MMRE = np.divide(np.abs(error), true).mean()

    RMSE = math.sqrt(np.square(np.subtract(true, predict)).mean())

    return R_squared, chi_squared, RMSE, MMRE



def conculate_mean(df):
    # This is for getting the mean of each parameter using all the galaxies
    processed_Df = df.groupby(['col1'], as_index= False)['col1'].mean()
    processed_Df['col2'] = df.groupby(['col1'], as_index= False)['col2'].mean()['col2']
    processed_Df['col3'] = df.groupby(['col1'], as_index= False)['col3'].mean()['col3']

    return processed_Df

def eval(dist_x2_given_x1, model, valid_loader = None):
  model.to(device)
  model.eval()
  
  predicted = np.array([])
  true = np.array([])
  error = np.array([])
  
  selectedProperty = np.array([])
  flowSamples = np.array([])
  
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
        
        # This is for extra analysis for property and error bar dependence, no entiendo mucho esto
        #pointer = 0
        #for index in range(data.note.shape[0]):
        #  Subhalos = data.x[pointer: (pointer + data.note[index])].detach().cpu().numpy()
        #  NdarkGalaxy = 0
        #  for subhalo in Subhalos:
        #    if subhalo[4] <= 1:
        #      NdarkGalaxy +=1
            # else:
              # print("Find a Dark Subhalo")
        #  selectedProperty = np.append(selectedProperty, NdarkGalaxy)
        #  pointer = pointer + data.note[index]
        
        
        true = np.append(true, y_target.detach().cpu().numpy())
        
        predicted = np.append(predicted, flowMean.detach().cpu().numpy())
        
        error = np.append(error, flowSTD.detach().cpu().numpy())
        
        # Attach flow samples
        if i == 0:
            flowSamples = flowSample.squeeze().detach().cpu().numpy().transpose()
        else:
            flowSamples = np.vstack((flowSamples, flowSample.squeeze().detach().cpu().numpy().transpose()), )
            
        # Select on special flow sample for one posterior plot
        if i == 10:
            oneSamples = flowSample.squeeze().detach().cpu().numpy().transpose()[10]
                
        i+=1
  
  plt.clf()
  
  print(len(true))
  print(true, flush=True)
  print(predicted, flush=True)
  idx = np.random.choice(np.arange(len(true)), 100, replace=False)
  np.savez(f_result + "results", true, predicted, error)
  
  #print(idx)
  df = pd.DataFrame({'col1': true, 'col2': predicted, 'col3': error})
  # df = conculate_mean(df)

  R_squared, chi_squared, RMSE, MMRE = calculate_metrics(true, predicted, error)
  
  true = true[idx]
  print(len(true))
  predicted = predicted[idx]
  error = error[idx]
#   randomly select
#   plt.scatter(true, predicted, s=1)
  # plt.errorbar(df['col1'], df['col2'], yerr=df['col3'], fmt='o', markersize=2, lw = 0.3)
  plt.errorbar(true, predicted, yerr=error, fmt='o', markersize=2, lw = 0.3)  
  plt.text(0.02, 0.45,
            f"$R^2$:{round(R_squared,3)}"
            "\n"
            rf"RMSE:{round(RMSE,3)}"
            "\n"
            rf"$\epsilon$:{round(MMRE,3)}"
            "\n"
             )
  xref = np.linspace(0.03,0.58,100)
  yref = xref
  plt.plot(xref, yref, color='red', linestyle='-')
  plt.xlabel(r'True $1/m_{WDM}$ [keV]')
  plt.ylabel(r'Pred $1/m_{WDM}$ [keV]')
  #plt.title('Deepset + NF: all node features unless MBH')
  plt.savefig(f_result + f"{study_name}sample" + ".png")

  # This is for one sample plot
  # plt.clf()
  # print(oneSamples.shape, flush=True)
  # sns.distplot(oneSamples, hist=False, kde=True,
  #       bins=None, color='firebrick',
  #       hist_kws={'edgecolor':'black'},
  #       kde_kws={'linewidth': 2},
  #       label='flow')
  # print(true[31], flush=True)
  # plt.grid()
  # plt.xlim(-0.1,0.6)

  # plt.savefig("results/" + f"{study_name}One" + ".pdf")
  
  # This is for error dependence plot
  # plt.clf()
  
  # plt.scatter(error, selectedProperty, s = 0.1)
  # plt.savefig(f"results/{study_name}ErrorNDarkGalaxy.pdf")
  
  # plt.clf()
  
  # flowSamples = flowSamples.transpose()[:,:,np.newaxis]
  # print(flowSamples.shape, flush=True)
  # print(true.shape, flush=True)
  # coverage = get_drp_coverage(flowSamples, true[:,np.newaxis])
  # plt.plot(np.array(coverage)[0],np.array(coverage)[1])
  # x = np.linspace(0,1,100)
  # plt.plot(x,x, linestyle='dashed')
  # plt.xlabel("Credibility Level")
  # plt.ylabel("Expected Coverage")
  # plt.savefig(f"results/{study_name}coverage.pdf")
  
  return 0

eval(dist_x2_given_x1, model, valid_loader)






