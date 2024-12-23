import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import optuna
import sys
import math

device = ""

if torch.cuda.is_available():
    print("CUDA Available")
    #device = torch.device('cuda:'+ sys.argv[1])
    device = torch.device('cuda:1')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


best_loss = 1000                       #set this to a large number. Used to compute
batch_size     = 32                        #number of elements each batch contains. Hyper-parameter
dr             = 0.0                       #dropout rate. Hyper-parameter
epochs         = 200                       #number of epochs to train the network. Hyper-parameter


model_name = '/model_MPL_mtot_datanorm_noomega_200/'
losses_name = '/losses_MPL_mtot_datanorm_noomega_200/'
study_name = 'MPL_mtot_datanorm_noomega_200'

torch.manual_seed(12345)

#################dataset#############
#mean_all = torch.load('mbh_mean_allsim_norm.pt')
#std_all = torch.load('std_mean_allsim_norm.pt')

#mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')
#omega = mass_sim[:,0]#.tolist()
#wdm_mass = mass_sim[:,-1].tolist()

data = np.load('mtot_mean_std.npy')


X_tensor = torch.tensor(data[:, :-2], dtype=torch.float32)
y_tensor = torch.tensor(data[:, -1], dtype=torch.float32)

X_mean = X_tensor.mean(dim=0)
print(X_mean)
X_std = X_tensor.std(dim=0)
X_tensor = (X_tensor - X_mean) / X_std

split_train = 720
split_valid = 870

input_train_tensor = X_tensor[0:split_train,:]
target_train_tensor = y_tensor[0:split_train]

print(np.shape(input_train_tensor))

input_valid_tensor = X_tensor[split_train:split_valid,:]
target_valid_tensor = y_tensor[split_train:split_valid]



#c_train = np.array([item for pair in zip(mbh_mean[0:split_train], std_mean[0:split_train], omega[0:split_train]) for item in pair])
#c_valid = np.array([item for pair in zip(mbh_mean[split_train:split_valid], std_mean[split_train:split_valid], omega[split_train:split_valid]) for item in pair])
#mean_all = np.array(mean_all)
#std_all = np.array(std_all)

#list_train = np.array([mean_all[0:split_train], std_all[0:split_train], omega[0:split_train]]) #mbh case
#list_train = np.concatenate([mean_all[0:split_train].reshape(split_train,1), std_all[0:split_train].reshape(split_train,1), omega[0:split_train].reshape(split_train,1)], axis=1)
#print(np.shape(list_train))
#input_train_tensor = torch.tensor(list_train, dtype=torch.float32)
#input_train_tensor = torch.tensor(list_train.T, dtype=torch.float32)
#target_train_tensor = torch.tensor(wdm_mass[0:split_train], dtype=torch.float32)

#list_valid = np.array([mean_all[split_train:split_valid], std_all[split_train:split_valid], omega[split_train:split_valid]]) #mbh case
#list_valid = np.concatenate([mean_all[split_train:split_valid].reshape(len(mean_all[split_train:split_valid]),1), std_all[split_train:split_valid].reshape(len(mean_all[split_train:split_valid]),1), omega[split_train:split_valid].reshape(len(mean_all[split_train:split_valid]),1)], axis=1)
#input_valid_tensor = torch.tensor(list_valid, dtype=torch.float32)

#input_valid_tensor = torch.tensor(list_valid.T, dtype=torch.float32)
#target_valid_tensor = torch.tensor(wdm_mass[split_train:split_valid], dtype=torch.float32)

##########################################

class CustomDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        # Devuelve el número de muestras, que es 2048 en este caso
        return len(self.input_data)

    def __getitem__(self, idx):
        # Devuelve la muestra correspondiente al índice `idx`
        return self.input_data[idx], self.target_data[idx]


class MPL(nn.Module):

    def __init__(self, input_size=3, output_size=1, num_layers=2, hidden_size=64):
        super(MPL, self).__init__()
        # Definimos las capas totalmente conectadas
        #self.fc1 = nn.Linear(input_size, hidden_size)  # Capa de entrada a capa oculta
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, output_size) # Capa oculta a capa de salida
        #self.relu = nn.ReLU()  # Función de activación ReLU

        #definir la primera capa
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        # capas ocultas
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Definir la capa de salida
        layers.append(nn.Linear(hidden_size, output_size))

        # Construir el modelo secuencial
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        # Aplicamos la capa completamente conectada con activación
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x)) 
        # Pasamos a la capa de salida
        #x = self.fc3(x)
        
        return self.model(x)


def name_model(hyperparameters):
    return "num_layers" + str(hyperparameters[0]) + "n_units_" + str(hyperparameters[1]) + "_lr_" + "{:.3e}".format(hyperparameters[2]) + "_wd_" + "{:.3e}".format(hyperparameters[3], 3) 


def train(model, optimizer, criterion, scheduler, train_loader):
  train_loss = 0.0
  model.train()
  
  for inputs, targets in train_loader:# Iterate in batches over the training dataset.
      inputs=inputs.to(device=device)
      targets=targets.to(device=device)
      optimizer.zero_grad()
      out = model(inputs)
      loss = criterion(out,targets.unsqueeze(1))
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      scheduler.step()
      train_loss += loss.item()
  last_loss = train_loss/len(train_loader)
  
  return last_loss

def eval(model, criterion, best_loss, valid_loader, hyperparameters):
  valid_loss = 0.0
  model.eval()
  
  for inputs, targets in valid_loader:# Iterate in batches over the training dataset.
      with torch.no_grad():
          inputs=inputs.to(device=device)
          targets=targets.to(device=device)
          out = model(inputs)
          loss = criterion(out,targets.unsqueeze(1))
          valid_loss += loss.item()
  val_loss = valid_loss/len(valid_loader)

  if val_loss < best_loss:
      print('saving')
      fmodel = model_name
      torch.save(model.state_dict(), '/home/bcostanza/MachineLearning/project/again' + model_name + name_model(hyperparameters))
      best_loss = val_loss
  
  return val_loss, best_loss


def objective(trial):
    # Create the dataset based on the link 
    #r_link = trial.suggest_float("r_link", 1.e-4, 1.e-2, log=True)
    train_dataset = CustomDataset(input_train_tensor, target_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = CustomDataset(input_valid_tensor, target_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    #suggest values in the number of neurons
    num_layers = trial.suggest_int("num_layers", 1, 5)
    n_units = trial.suggest_categorical("n_units", [32, 64, 128, 256])

    #input_size = 28 #if we use all features
    input_size = 2
    
    model = MPL(input_size=input_size, output_size=1, num_layers=num_layers, hidden_size=n_units)
        
    criterion = nn.MSELoss()  #loss function. In this case MSE (mean squared error)
    
    lr = trial.suggest_float('lr', 1e-8,1e-5, log=True)
    #lr = trial.suggest_float('lr', 1e-5,1e-1)
    wd = trial.suggest_float('wd', 1e-9,1e-6, log=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1.e-4, cycle_momentum=False, step_size_up=1000)

    model.to(device=device)

    trainLoss_history = []
    validLoss_history = []

    best_loss = 1e7
    print('starting with:',best_loss)

    for epoch in range(epochs):
        print(epoch, flush= True) #me printea las epocas en cada run
        train_loss = train(model,optimizer,criterion, scheduler, train_loader)
        valid_loss, best_loss = eval(model, criterion, best_loss, valid_loader, hyperparameters = [num_layers, n_units, lr, wd])

        print(f"train loss {train_loss}, valid loss {valid_loss}", flush = True)
        
        if(math.isnan(train_loss)):
            return 10000
        
        trainLoss_history.append(train_loss)
        validLoss_history.append(valid_loss)
    
    np.savez('/home/bcostanza/MachineLearning/project/again' + losses_name + name_model([num_layers, n_units, lr, wd]), trainLoss_history, validLoss_history)

    return best_loss


storage = f"sqlite:///{study_name}.db"
study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage, load_if_exists= True)
study.optimize(objective, n_trials=100)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))


print("Best trial:")
trial = study.best_trial

print("value:", trial.value)

print("Params:")
for key, value in trial.params.items():
	print("{}:{}".format(key, value))




