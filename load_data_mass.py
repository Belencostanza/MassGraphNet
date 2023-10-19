import os
import torch

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import numpy as np
import h5py

#functions for visualization
#%matplotlib inline
#import networkx as nx
#import matplotlib.pyplot as plt

from torch_geometric.data import Data
import numpy as np
from torch_geometric.loader import DataLoader

from multiprocessing import Pool
from copy import deepcopy

n_CPU = 40

#number of star threshold
Nstar_th = 5

#Use cosmological parameters as global features
global_parameters = True

#in this case I'm creating a training dataset and a validation dataset splitting the simulations
simpathroot = '/mnt/home/bwang/MassGraphNet/data'
nsim = 1024

#nhalos = 20 #number of haloes per simulation
#put numbers to choose how many simulations are you taking for training, validation and testing
split_train = 720  #0 to 500 simulations for training
split_valid = 870 #500 to 700 simulations for validation
split_test = nsim #700 to 750 simulations for testing

#name of the dataset 
name_train = 'masswdm_train_menos10_all.pt'
name_valid = 'masswdm_valid_menos10_all.pt'
name_test = 'masswdm_test_menos10_all.pt'

#--------------------------------------------------------------------------------------------------------

def correct_boundary(pos, boxlength=1.):
    
    for i, pos_i in enumerate(pos):
        for j, coord in enumerate(pos_i):
            if coord > boxlength/2.:
                pos[i,j] -= boxlength
            elif -coord > boxlength/2.:
                pos[i,j] += boxlength
    return pos

def euclidean_distance(n_sub,position):
    distance = torch.zeros((n_sub,n_sub))
    for k in range(len(position)):
        for j in range(len(position)):
              distance[k,j] = torch.cdist(position[k].unsqueeze(0), position[j].unsqueeze(0), p=2)
    return distance

def features_new(fin):
    
    f = h5py.File(fin, 'r')
    
    header = f['Header']
    boxsize = header.attrs[u'BoxSize']
    
    #load subhalos features
    Pos_subhalo = f["Subhalo/SubhaloPos"][:]/boxsize
    Mstar = f["Subhalo/SubhaloMassType"][:,4]  #total mass of all members particles
    Mdm = f["Subhalo/SubhaloMassType"][:,1] 
    Nstar = f["Subhalo/SubhaloLenType"][:,4]
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4] #radnorm #radius galaxy
    GMetal = f["Subhalo/SubhaloGasMetallicity"][:] #gas metallicity
    SMetal = f["Subhalo/SubhaloStarMetallicity"][:] #star metallicity
    Vmax = f["Subhalo/SubhaloVmax"][:] 

    Vel_subhalo = f["Subhalo/SubhaloVel"][:] #/velnorm
    HaloID = np.array(f["Subhalo/SubhaloGrNr"][:], dtype=np.int32)  #It tells you to which halo belongs every subhalo

    # Load halo features
    HaloMass = f["Group/GroupMass"][:]
    Pos_halo = f["Group/GroupPos"][:]/boxsize
    Vel_halo = f["Group/GroupVel"][:] #/velnorm 
    
    f.close()

    # Neglect halos with zero mass
    indexes = np.argwhere(HaloMass>0.).reshape(-1) #haloes index in the given simulation
    
    #threshold in the number of stars. 
    #indexes_star = np.where(Nstar>Nstar_th)[0]
    
    #Pos_subhalo = Pos_subhalo[indexes_star] 
    #Mstar   = Mstar[indexes_star]
    #Rstar   = Rstar[indexes_star]
    #Mdm = Mdm[indexes_star]
    #HaloID = HaloID[indexes_star]
    #Vel_subhalo = Vel_subhalo[indexes_star]
    #GMetal = GMetal[indexes_star]
    #SMetal = SMetal[indexes_star]
    #Vmax = Vmax[indexes_star]
    
    #correct simulations outside the box
    Pos_subhalo[np.where(Pos_subhalo<0.0)]+=1.0
    Pos_subhalo[np.where(Pos_subhalo>1.0)]-=1.0
    
    #take the logarithm
    Mstar = np.log10(1.+Mstar)
    Mdm = np.log10(1.+Mdm)
    Rstar = np.log10(1.+Rstar)
    GMetal = np.log10(1.+GMetal)
    SMetal = np.log10(1.+SMetal)
    Vmax = np.log10(1.+Vmax)
    
    #normalize the variables
    Vel_subhalo = normalize(Vel_subhalo)
    Vel_halo = normalize(Vel_halo)
    
    Mstar = normalize(Mstar)
    Mdm = normalize(Mdm)
    Rstar = normalize(Rstar)
    GMetal = normalize(GMetal)
    SMetal = normalize(SMetal)
    Vmax = normalize(Vmax)
    
    #1. position
    #2. star mass
    #3. Radio 
    #4. dark matter mass 
    #5. Gas Metallicity
    #6. Star Metallicity 
    #7. Vmax
    #8. subhalo velocity
    tab = np.column_stack((HaloID, Pos_subhalo, Mstar, Rstar, Mdm, GMetal, SMetal, Vmax, Vel_subhalo))
    #tab_features = np.column_stack((Mstar,Rstar,Mdm,Metal))
    #x = torch.tensor(tab_features, dtype=torch.float32)

    
    return tab, Pos_halo, Vel_halo, indexes
 
def normalize(variable):
    mean, std = variable.mean(axis=0), variable.std(axis=0)
    norm_variable = (variable - mean)/std
    return norm_variable
    

def create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters, r_link = 1e-1):
    
    #this for goes through every halo inside the simulation
    #num_sub = []
    data_sim = []
    for ind in halolist: 
        n_sub = len(tab[tab[:,0]==ind])
        #num_sub.append(n_sub) 
    
        if n_sub < 10 and n_sub > 4: 
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            
            #tab_halo[:,0:3] -= GroupPos[ind]  #in the halo frame
            #tab_halo[:,-3:] -= GroupVel[ind]  
            
            distance = euclidean_distance(n_sub, torch.Tensor(tab_halo[:,0:3])) 
            index_mask = (distance > 0) & (distance < r_link)
            index_edge = np.array(np.where(index_mask == True))
            index_edge = torch.tensor(index_edge, dtype=torch.long)
            
            edge_attr = torch.zeros((index_edge.shape[1], 1)) #shape=[number of edges, features=0]
            
            u_number = np.log10(n_sub).reshape(1,1) #number of subhalos in the simulation as a global feature
            #print(np.shape(u_number))           

            if global_parameters == True:
                u_parameters = parameters.reshape(1,5)
                #print(np.shape(u_parameters))
                u = np.concatenate((u_number, u_parameters), axis=1)
            else:
                u = u_number
                
            mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
            data = Data(x=torch.Tensor(tab_halo), u = torch.tensor(u, dtype=torch.float32), edge_index = index_edge, edge_attr = edge_attr, y=mass)
            data_sim.append(data)
            
    return data_sim

#-------------------------------------------------------------------------------------------------------------------------------

mass_sim = np.loadtxt('/mnt/home/bwang/MassGraphNet/data/sobol_sequence_WDM_real_values.txt')

#read the data
def create_start_end_indexes(start, end, number):
    
    step = (end-start)//number
    
    start_indexes = np.arange(start, end, step)
    end_indexes = np.arange(start+step, end+step, step)
    
    end_indexes[-1] = end
    
    return start_indexes, end_indexes

def create_ranged_graphs(index_start, index_end, r_link = 1e-2):
    
    print("Creating ranged graphs", flush = True)
    dataset = []
    for i in range(index_start, index_end):
        print('reading simulation', i, flush = True)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, GroupPos, GroupVel, indexes = features_new(fin)
        
        mwdm = mass_sim[i,-1]        
        parameters = mass_sim[i,:-1]  #other parameters of the simulation
    
        halolist = indexes#[:nhalos]
    
        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters, r_link)
    
        dataset += data_sim

    # Save the data to avoid too many file descriptor and receive 0 item issue

    torch.save(dataset, "./prepared_data/dataset_%d_%d.pt"%(index_start, index_end))
    print("finished saving dataset_%d_%d.pt"%(index_start, index_end), flush = True)
    


def training_set(r_link = 1e-1):
    dataset_train = []
    start_indexes, end_indexes = create_start_end_indexes(0, split_train, n_CPU)
    
    print(start_indexes, flush = True)
    print(end_indexes, flush = True)
        
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])
    
    print("finished training set multiprocessing part")

    for i in range(start_indexes.shape[0]):
        dataset_train += torch.load("./prepared_data/dataset_%d_%d.pt"%(start_indexes[i], end_indexes[i]))
    
    return dataset_train
    
def validation_set(r_link = 1e-1):
    dataset_valid = []
    start_indexes, end_indexes = create_start_end_indexes(split_train, split_valid, n_CPU)
    
    print(start_indexes, flush = True)
    print(end_indexes, flush = True)
    print(start_indexes.shape[0], flush = True)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])
        
    for i in range(start_indexes.shape[0]):
        dataset_valid += torch.load("./prepared_data/dataset_%d_%d.pt"%(start_indexes[i], end_indexes[i]))

    return dataset_valid

def test_set(r_link = 1e-1):
    dataset_test = []
    start_indexes, end_indexes = create_start_end_indexes(split_valid, split_test, n_CPU)
    
    print(start_indexes, flush = True)
    print(end_indexes, flush = True)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])

    for i in range(start_indexes.shape[0]):
        dataset_test += torch.load("./prepared_data/dataset_%d_%d.pt"%(start_indexes[i], end_indexes[i]))
    
    return dataset_test


if __name__ == "__main__":
    print('reading')

    train_dataset = training_set()
    torch.save(train_dataset, name_train)

    valid_dataset = validation_set()
    torch.save(valid_dataset, name_valid)
        
    test_dataset = test_set()
    torch.save(test_dataset, name_test)





