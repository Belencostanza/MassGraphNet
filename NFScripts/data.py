import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

root = "/mnt/home/bwang/NFDM/data/"

def loadCondition():
  # input data
  X = []
  for i in range(2000):
    path = root + "abundance_z=0.0_" + str(i) + "_HR_linspace_60_5_19bins_untrimmed.csv"
    if not os.path.exists(path):
      continue
    f = pd.read_csv(path)
    X.append(f["densitycontrast"])

  X = np.array(X)
  X = StandardScaler().fit_transform(X)

  return X

def loadTarget():
  # Since output took much less time to load, it is seperate function
  # output data
  params = []
  f_para = pd.read_csv(root +"latin_hypercube_params.txt", sep = " ", header = None)

  for i in range(2000):
    path = root + "abundance_z=0.0_" + str(i) + "_HR_linspace_60_5_19bins_untrimmed.csv"
    if not os.path.exists(path):
      continue
    params.append(np.array(f_para)[i])
    
  params = np.array(params)
  params_Scaler = StandardScaler()
  params = params_Scaler.fit_transform(params)

  return params, params_Scaler

def prepare_dataSet(input, output, device):

  # Split into training validation and test
  X = torch.tensor(input, dtype=torch.float, device=device)
  Y = torch.tensor(output, dtype=torch.float, device=device)

  training_size = int(input.shape[0]*0.7)
  validation_size = int(input.shape[0]*0.15)
  test_size = input.shape[0] - training_size - validation_size

  input_training, input_valid, input_test = torch.utils.data.random_split(X, [training_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
  output_training, output_valid, output_test = torch.utils.data.random_split(Y, [training_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

  # This is nasty but it works
  input_test = torch.stack([d for d in input_test])
  input_valid = torch.stack([d for d in input_valid])
  input_training = torch.stack([d for d in input_training])
  output_test = torch.stack([d for d in output_test])
  output_valid = torch.stack([d for d in output_valid])
  output_training = torch.stack([d for d in output_training])

  return input_training, input_valid, input_test, output_training, output_valid, output_test