import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

def create_condDist(targe_dimension, context_dimension, n_conditional_layers, count_bins, hidden_dims, device):
    
  conditional_transforms = []
  
  dist_base2 = dist.Normal(torch.zeros(targe_dimension, device = device), torch.ones(targe_dimension, device = device), validate_args=False)
  
  for i in range(n_conditional_layers):
    conditional_transforms.append(T.conditional_spline_autoregressive(targe_dimension, context_dim=context_dimension, bound=5, count_bins=count_bins, hidden_dims=hidden_dims))
 
  dist_x2_given_x1 = dist.ConditionalTransformedDistribution(dist_base2, conditional_transforms)

  return dist_x2_given_x1, conditional_transforms

def createAchi(transforms, lr, wd, device, fromLoad = False):
  
  modules = torch.nn.ModuleList(transforms)
  modules.to(device)
  optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay = wd)

  if fromLoad:
    print("Loading from existing model...")
    # modules = torch.load(‘modules.pt’)
    # x1_transform = torch.load(‘x1_transform.pt’)
    # modules.load_state_dict(torch.load(“modules_state_dict.pt”))

  return modules, optimizer
