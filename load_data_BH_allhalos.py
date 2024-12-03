import os
import torch


os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import numpy as np
import h5py
 
from torch_geometric.data import Data
import numpy as np
from torch_geometric.loader import DataLoader

from multiprocessing import Pool
import scipy.spatial as SS

import sys
import random

device = ""

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


n_CPU = 10


#in this case I'm creating a training dataset and a validation dataset splitting the simulations
#simpathroot = '/hildafs/projects/phy200026p/bwangc/MassGraphNet/data/users.flatironinstitute.org/~fvillaescusa/VOSS5'
simpathroot = '/data/bcostanza/data'
nsim = 1024

#nhalos = 20 #number of haloes per simulation
#put numbers to choose how many simulations are you taking for training, validation and testing
split_train = 720  #0 to 500 simulations for training
split_valid = 870 #500 to 700 simulations for validation
split_test = nsim #700 to 750 simulations for testing


#study_name = "BlackHoleDF4TO10"
study_name = "BlackHoleDFALLhalos_norm_global"


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

def features_new_v2(fin, mean=0., std=1., norm=False):

    f = h5py.File(fin, 'r')

    header = f['Header']
    boxsize = header.attrs[u'BoxSize']

    #load subhalos features
    Pos_subhalo = f["Subhalo/SubhaloPos"][:]/boxsize #I only use this for the r_link calculation

    Mbh  = f['/Subhalo/SubhaloBHMass'][:]*1e10 #black hole 
    HaloMass = f["Group/GroupMass"][:]

    HaloID = np.array(f["Subhalo/SubhaloGrNr"][:], dtype=np.int32)  #It tells you to which halo belongs every subhalo

    f.close()

    Mbh = np.log10(1.+Mbh)
    #Mbh = normalize(Mbh) #quitamos normalizacion por simulación
    mean_mbh, std_mbh = normalize(Mbh)
    if norm==True: 
        Mbh = (Mbh - mean)/std


    tab = np.column_stack((HaloID,Mbh))

    indexes = np.argwhere(HaloMass>0.).reshape(-1) #haloes index in the given simulation

    return tab, indexes, mean_mbh, std_mbh
 
def normalize(variable):
    mean, std = variable.mean(axis=0), variable.std(axis=0)
    #norm_variable = (variable - mean)/std
    return mean, std

def variables(halolist, tab):
    all_features = []
    for ind in halolist:
        n_sub = len(tab[tab[:,0]==ind])
        #num_sub.append(n_sub)

        #if n_sub < 10 and n_sub > 4:
        if n_sub > 4: # if you want to run the condition of more than 4 subhalos uncomment this and comment the condition above
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            tab_features = tab_halo[:,:] #(nsub, features)
            all_features.append(tab_features)

    return all_features

    
def create_graphs_new(halolist, tab, mwdm, parameters, mean_mbh, std_mbh):
    
    all_tab_features = []
    for ind in halolist: 
        n_sub = len(tab[tab[:,0]==ind])
        if n_sub > 4:
            #print(ind, n_sub)
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            tab_feat = tab_halo[:,:]  #(nsub,features)
            all_tab_features.append(tab_feat)

    join_all = np.concatenate((all_tab_features), axis=0)

                        
    u_parameters = parameters[0:1]
    u_parameters = u_parameters.reshape(1,1)
    u_mean = mean_mbh.reshape(1,1)
    u_std = std_mbh.reshape(1,1)
    u = np.concatenate((u_parameters, u_mean, u_std), axis=1)
                         
    mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
    #print(all_tab_features[1], flush = True)
    
    data = Data(x=torch.Tensor(np.array(join_all)), u = torch.tensor(u, dtype=torch.float32), y=mass)
    
    return [data]

#-------------------------------------------------------------------------------------------------------------------------------

mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')
#mass_sim = np.loadtxt('/hildafs/projects/phy200026p/bwangc/MassGraphNet/data/users.flatironinstitute.org/~fvillaescusa/VOSS5/sobol_sequence_WDM_real_values.txt')

#read the data
def create_start_end_indexes(start, end, number):
    
    step = (end-start)//number
    
    start_indexes = np.arange(start, end, step)
    end_indexes = np.arange(start+step, end+step, step)
    
    end_indexes[-1] = end
    
    return start_indexes, end_indexes

def create_ranged_graphs(index_start, index_end, mean, std, r_link = 1e-2):
    
    # print("Creating ranged graphs", flush = True)
    dataset = []
    for i in range(index_start, index_end):
        # print('reading simulation', i, flush = True)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, indexes, mean_mbh, std_mbh = features_new_v2(fin, mean, std, norm=True)
        
        mwdm = mass_sim[i,-1]        
        parameters = mass_sim[i,:-1]  #other parameters of the simulation
    
        halolist = indexes#[:nhalos]
    
        data_sim = create_graphs_new(halolist, tab, mwdm, parameters, mean_mbh, std_mbh)
    
        dataset += data_sim

    # Save the data to avoid too many file descriptor and receive 0 item issue

    #torch.save(dataset, "./prepared_data/dataset_%d_%d_%s_%s.pt"%(index_start, index_end, study_name, sys.argv[1]))
    torch.save(dataset, "./prepared_data_norm_global/dataset_%d_%d_%s.pt"%(index_start, index_end, study_name))

    # print("finished saving dataset_%d_%d.pt"%(index_start, index_end), flush = True)
    

def training_set(r_link = 1e-1):

    features_all = []
    for i in range(0,split_train):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        tab, indexes,_,_ = features_new_v2(fin)
        variables_all = variables(indexes, tab)

        features_all += variables_all

    features_all = np.array(np.concatenate(features_all, axis=0))
    mean = np.mean(features_all)
    std = np.std(features_all)

    dataset_train = []
    start_indexes, end_indexes = create_start_end_indexes(0, split_train, n_CPU)    
        
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], mean, std, r_link) for i in range(start_indexes.shape[0])])
    
    print("finished training set multiprocessing part")

    for i in range(start_indexes.shape[0]):
        #dataset_train += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_train += torch.load("./prepared_data_norm_global/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))
    
    return dataset_train
    
def validation_set(r_link = 1e-1):

    features_all = []
    for i in range(split_train,split_valid):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        tab, indexes,_,_ = features_new_v2(fin)
        variables_all = variables(indexes, tab)

        features_all += variables_all

    features_all = np.array(np.concatenate(features_all, axis=0))
    mean = np.mean(features_all)
    std = np.std(features_all)

    dataset_valid = []
    start_indexes, end_indexes = create_start_end_indexes(split_train, split_valid, n_CPU)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], mean, std, r_link) for i in range(start_indexes.shape[0])])
        
    for i in range(start_indexes.shape[0]):
        #dataset_valid += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_valid += torch.load("./prepared_data_norm_global/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))


    return dataset_valid

def test_set(r_link = 1e-1):

    features_all = []
    for i in range(split_valid,split_test):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        tab, indexes,_,_ = features_new_v2(fin)
        variables_all = variables(indexes, tab)

        features_all += variables_all

    features_all = np.array(np.concatenate(features_all, axis=0))
    mean = np.mean(features_all)
    std = np.std(features_all)

    dataset_test = []
    start_indexes, end_indexes = create_start_end_indexes(split_valid, split_test, n_CPU)
    
    #print(start_indexes, flush = True)
    #print(end_indexes, flush = True)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], mean, std, r_link) for i in range(start_indexes.shape[0])])

    for i in range(start_indexes.shape[0]):
        #dataset_test += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_test += torch.load("./prepared_data_norm_global/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))

    
    return dataset_test


if __name__ == "__main__":
    print('reading')

    train_dataset = training_set()
    # torch.save(train_dataset, name_train)

    valid_dataset = validation_set()
    # torch.save(valid_dataset, name_valid)
        
    test_dataset = test_set()
    # torch.save(test_dataset, name_test)





