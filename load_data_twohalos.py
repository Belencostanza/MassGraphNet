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


n_CPU = 20


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
study_name = "BlackHoleDF2halos"


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

def features_new_v2(fin):

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
    Mbh = normalize(Mbh)    

    tab = np.column_stack((HaloID,Mbh))

    indexes = np.argwhere(HaloMass>0.).reshape(-1) #haloes index in the given simulation

    return tab, indexes
 
def normalize(variable):
    mean, std = variable.mean(axis=0), variable.std(axis=0)
    norm_variable = (variable - mean)/std
    return norm_variable
    

def create_graphs_new(halolist, tab, mwdm, parameters):
    
    all_tab_features = []
    for ind in halolist: 
        n_sub = len(tab[tab[:,0]==ind])
        if n_sub > 4:
            #print(ind, n_sub)
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            tab_feat = tab_halo[:,:]  #(nsub,features)
            all_tab_features.append(tab_feat)

    all_two = random.sample(all_tab_features, 2)
    join_two = np.concatenate((all_two[0], all_two[1]), axis=0)

                        
    u_parameters = parameters[0:1]
    u = u_parameters.reshape(1,1)
                         
    mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
    #print(all_tab_features[1], flush = True)
    
    data = Data(x=torch.Tensor(np.array(join_two)), u = torch.tensor(u, dtype=torch.float32), y=mass)
    
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

def create_ranged_graphs(index_start, index_end, r_link = 1e-2):
    
    # print("Creating ranged graphs", flush = True)
    dataset = []
    for i in range(index_start, index_end):
        # print('reading simulation', i, flush = True)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]        
        parameters = mass_sim[i,:-1]  #other parameters of the simulation
    
        halolist = indexes#[:nhalos]
    
        data_sim = create_graphs_new(halolist, tab, mwdm, parameters)
    
        dataset += data_sim

    # Save the data to avoid too many file descriptor and receive 0 item issue

    #torch.save(dataset, "./prepared_data/dataset_%d_%d_%s_%s.pt"%(index_start, index_end, study_name, sys.argv[1]))
    torch.save(dataset, "./prepared_data/dataset_%d_%d_%s.pt"%(index_start, index_end, study_name))

    # print("finished saving dataset_%d_%d.pt"%(index_start, index_end), flush = True)
    


def training_set(r_link = 1e-1):
    dataset_train = []
    start_indexes, end_indexes = create_start_end_indexes(0, split_train, n_CPU)
    
        
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])
    
    print("finished training set multiprocessing part")

    for i in range(start_indexes.shape[0]):
        #dataset_train += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_train += torch.load("./prepared_data/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))

    
    return dataset_train
    
def validation_set(r_link = 1e-1):
    dataset_valid = []
    start_indexes, end_indexes = create_start_end_indexes(split_train, split_valid, n_CPU)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])
        
    for i in range(start_indexes.shape[0]):
        #dataset_valid += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_valid += torch.load("./prepared_data/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))


    return dataset_valid

def test_set(r_link = 1e-1):
    dataset_test = []
    start_indexes, end_indexes = create_start_end_indexes(split_valid, split_test, n_CPU)
    
    print(start_indexes, flush = True)
    print(end_indexes, flush = True)
    
    with Pool(start_indexes.shape[0]) as p:
        p.starmap(create_ranged_graphs, [(start_indexes[i], end_indexes[i], r_link) for i in range(start_indexes.shape[0])])

    for i in range(start_indexes.shape[0]):
        #dataset_test += torch.load("./prepared_data/dataset_%d_%d_%s_%s.pt"%(start_indexes[i], end_indexes[i], study_name, sys.argv[1]))
        dataset_test += torch.load("./prepared_data/dataset_%d_%d_%s.pt"%(start_indexes[i], end_indexes[i], study_name))

    
    return dataset_test


if __name__ == "__main__":
    print('reading')

    train_dataset = training_set()
    # torch.save(train_dataset, name_train)

    valid_dataset = validation_set()
    # torch.save(valid_dataset, name_valid)
        
    test_dataset = test_set()
    # torch.save(test_dataset, name_test)





