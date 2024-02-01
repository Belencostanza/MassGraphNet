import optuna
import torch
from scripts.init import device
from scripts.archi import createAchi, create_condDist

class Objective(object):
  def __init__(self, input_training, input_valid, output_training, output_valid):
    self.input_training = input_training
    self.input_valid = input_valid
    self.output_training = output_training
    self.output_valid = output_valid
  def __call__(self, trial):

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    condition_layers = trial.suggest_int("condition_layers", 1,15)
    wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
    count_bins = trial.suggest_int("count_bins", 4, 256)
    
    # TODO:To choose from a list
    hidden = trial.suggest_int("hidden_dims", 0, 4)

    dim_options = [16,32,64,128,256]
    hidden_dims = [dim_options[hidden],dim_options[hidden]]
    
    steps = 200

    dist_x2_given_x1, x2_transform = create_condDist(5,18,condition_layers, count_bins,hidden_dims ,device)

    modules, optimizer = createAchi(x2_transform, lr, wd, device)
    
    min_valid = 100000
    
    for step in range(steps):
      optimizer.zero_grad()
      ln_p_x2_given_x1 = dist_x2_given_x1.condition(self.input_training).log_prob(self.output_training)
      loss = -(ln_p_x2_given_x1).mean()
      loss.backward()
      optimizer.step()
      dist_x2_given_x1.clear_cache()  
      
      fmodel = f"/mnt/home/bwang/NFDM/models/"
      
      # if validation loss is smaller then save the model
      # detach the tensor so that it does not require grad
      
      with torch.no_grad():
        ln_p_x2_given_x1 = dist_x2_given_x1.condition(self.input_valid.detach()).log_prob(self.output_valid.detach())
        valid_loss = -(ln_p_x2_given_x1).mean()
        
        if min_valid > valid_loss:
          torch.save(dist_x2_given_x1, fmodel+f"dist_x2_given_x1_{trial.number}.pt")
          torch.save(modules, fmodel+f"modules_{trial.number}.pt")
          torch.save(modules.state_dict(), fmodel+f"modules_state_dict_{trial.number}.pt")
          min_valid = valid_loss
          
      #if step % 10 == 0:
       # print(f"step:{step}, loss:{loss}, valid_loss:{valid_loss}")    
        
    return valid_loss
