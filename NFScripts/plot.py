import seaborn as sns
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import os


# TODO: here add save fig after all the plots are shown

def getRSquare_Coefficent(predicted, expected):
  RSS =np.square(np.subtract(expected, predicted)).sum();
  TSS = np.square(np.subtract(expected, np.mean(expected))).sum();

  RSquared = 1 - RSS/TSS;

  return RSquared;
def getChiValue(predicted, expected, error):
    sum = 0;
    size = len(predicted)
    for i in range(len(expected)):
        differance = predicted[i] - expected[i];
        squared = math.pow(differance,2);
        error_Divided = squared/math.pow(error[i],2);
        sum += error_Divided;

    chiValue = sum/size;


    return chiValue;

def show_onePostior(input_data, n_simulation, distribution, output_index,scaler = None):
  flow = distribution.condition(input_data[n_simulation]).sample(torch.Size([1000,]))
  if scaler != None:
    flow =  scaler.inverse_transform(flow)
  sns.distplot(flow[:,output_index], hist=False, kde=True,
              bins=None, color='firebrick',
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2},
              label='flow')
  plt.grid()
  
def show_conture(input_data, output_data, n_simulation, distribution, scaler = None,):
  flow = distribution.condition(input_data[n_simulation]).sample(torch.Size([1000,]))
  if scaler != None:
    flow = scaler.inverse_transform(flow)
    output_data = scaler.inverse_transform(output_data)
    
  sns.set(style='dark',)    
  res = sns.kdeplot(x = flow[:,0], y =flow[:,1],cmap="Purples_d", cbar=True, levels=[0.05,0.34], common_norm = False)
  plt.scatter(output_data[n_simulation][0],output_data[n_simulation][1], marker = 'x', c= 'red')

  padding = (output_data.max() - output_data.min())*0.1
  plt.xlim(output_data[:,0].min()- padding, output_data[:,0].max() + padding)
  plt.ylim(output_data[:,1].min()- padding, output_data[:,1].max() + padding)
  plt.show()
  
def plot_mean_std(results, output_valid, Y_Scaler):
  indexes = np.random.choice(np.arange(results.shape[0]),100,replace=False)
  
  true = Y_Scaler.inverse_transform(output_valid)[:,0]
  predicted = results[:,0]
  std = results[:,1]
  
  plt.errorbar(true[indexes],predicted[indexes],std[indexes], fmt="o")
  plt.plot(np.linspace(0.1,0.5,100),np.linspace(0.1,0.5,100))
  
  plt.text(0.42,0.1,"R^2: "+ str(round(getRSquare_Coefficent(predicted, true),3)))
  plt.text(0.42,0.08, "Chi^2: "+ str(round(getChiValue(predicted, true, std), 3)))
  
  plt.savefig(os.getcwd()+"/plots/mean_std.png")