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

n_CPU = 10

#number of star threshold
Nstars_th = 1

#Use cosmological parameters as global features
global_parameters = False
cosmo_parameters = True  #use only cosmological parameters
astro_parameters = False #use only astrophysical parameters

#in this case I'm creating a training dataset and a validation dataset splitting the simulations
#simpathroot = '/mnt/home/bwang/MassGraphNet/data'
simpathroot = '/data/bcostanza/data'
nsim = 1024

#nhalos = 20 #number of haloes per simulation
#put numbers to choose how many simulations are you taking for training, validation and testing
split_train = 720  #0 to 500 simulations for training
split_valid = 870 #500 to 700 simulations for validation
split_test = nsim #700 to 750 simulations for testing

#name of the dataset 
#name_train = 'masswdm_train_menos10_all_bonny.pt'
#name_valid = 'masswdm_valid_menos10_all_bonny.pt'
#name_test = 'masswdm_test_menos10_all_bonny.pt'

#--------------------------------------------------------------------------------------------------------


# Edge feature dierectly from cosmograph net
def get_edges(pos, r_link, use_loops):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r_link
    # Boxsize normalize to 1
    kd_tree = SS.KDTree(pos, leafsize=16, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.T
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes

    row, col = edge_index
    diff = pos[row]-pos[col]

    # Take into account periodic boundary conditions, correcting the distances
    for i, pos_i in enumerate(diff):
        for j, coord in enumerate(pos_i):
            if coord > r_link:
                diff[i,j] -= 1.  # Boxsize normalize to 1
            elif -coord > r_link:
                diff[i,j] += 1.  # Boxsize normalize to 1

    # Get translational and rotational invariant features
    
    # Distance
    dist = np.linalg.norm(diff, axis=1)
    
    # Centroid of galaxy catalogue
    centroid = np.mean(pos,axis=0)

    #Vectors of node and neighbor
    row = (pos[row] - centroid)
    col = (pos[col] - centroid)

   # Take into account periodic boundary conditions: row and col
    for i, pos_i in enumerate(row):
        for j, coord in enumerate(pos_i):
            if coord > 0.5:
                row[i,j] -= 1.  # Boxsize normalize to 1
                
            elif -coord > 0.5:
                row[i,j] += 1.  # Boxsize normalize to 1                                                

    for i, pos_i in enumerate(col):
        for j, coord in enumerate(pos_i):
            if coord > 0.5:
                col[i,j] -= 1.  # Boxsize normalize to 1
                
            elif -coord > 0.5:
                col[i,j] += 1.  # Boxsize normalize to 1
                
    # Normalizing
    unitrow = row/np.linalg.norm(row, axis = 1).reshape(-1, 1)
    unitcol = col/np.linalg.norm(col, axis = 1).reshape(-1, 1)
    unitdiff = diff/dist.reshape(-1,1)
    
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

    # Normalize distance by linking radius
    dist /= r_link

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    # Add loops
    if use_loops:
        loops = np.zeros((2,pos.shape[0]),dtype=int)
        atrloops = np.zeros((pos.shape[0],3))
        for i, posit in enumerate(pos):
            loops[0,i], loops[1,i] = i, i
            atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
        edge_index = np.append(edge_index, loops, 1)
        edge_attr = np.append(edge_attr, atrloops, 0)
    edge_index = edge_index.astype(int)

    return edge_index, edge_attr

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
    Mg = f['/Subhalo/SubhaloMassType'][:,0]*1e10 #gass mass content (Msun/h)
    Mstar = f["Subhalo/SubhaloMassType"][:,4]*1e10  #stellar mass of the galaxy
    Mbh  = f['/Subhalo/SubhaloBHMass'][:]*1e10 #black hole mass of the galaxy
    Mtot = f['/Subhalo/SubhaloMass'][:]*1e10 #total mass of the subhalo hosting the galaxy

    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]/1e3 #Mpc/h #radnorm #radius galaxy
    Rtot = f['/Subhalo/SubhaloHalfmassRad'][:]/1e3 #Mpc/h
    Rvmax  = f['/Subhalo/SubhaloVmaxRad'][:]/1e3 #Mpc/h


    GMetal = f["Subhalo/SubhaloGasMetallicity"][:] #gas metallicity
    SMetal = f["Subhalo/SubhaloStarMetallicity"][:] #star metallicity

    Vmax = f["Subhalo/SubhaloVmax"][:] 
    Vdisp  = f['/Subhalo/SubhaloVelDisp'][:]
    SFR = f['/Subhalo/SubhaloSFR'][:] #star formation rate
    J  = f['/Subhalo/SubhaloSpin'][:] #subhalo spin
    V  = f['/Subhalo/SubhaloVel'][:]
    J  = np.sqrt(J[:,0]**2 + J[:,1]**2 + J[:,2]**2)
    V  = np.sqrt(V[:,0]**2 + V[:,1]**2 + V[:,2]**2)

    Nstars = f['/Subhalo/SubhaloLenType'][:,4] #comment if you don't want a threshold in the number of stars

    #U = f['/Subhalo/SubhaloStellarPhotometrics'][:,0]
    #K = f['/Subhalo/SubhaloStellarPhotometrics'][:,3]
    #g = f['/Subhalo/SubhaloStellarPhotometrics'][:,4]

    #Vel_subhalo = f["Subhalo/SubhaloVel"][:] #/velnorm
    HaloID = np.array(f["Subhalo/SubhaloGrNr"][:], dtype=np.int32)  #It tells you to which halo belongs every subhalo

    # Load halo features
    HaloMass = f["Group/GroupMass"][:]
    Pos_halo = f["Group/GroupPos"][:]/boxsize
    Vel_halo = f["Group/GroupVel"][:] #/velnorm 

    f.close()

    # Neglect halos with zero mass
    indexes = np.argwhere(HaloMass>0.).reshape(-1) #haloes index in the given simulation


    #correct simulations outside the box -----> not necessary
    #Pos_subhalo[np.where(Pos_subhalo<0.0)]+=1.0
    #Pos_subhalo[np.where(Pos_subhalo>1.0)]-=1.0

    #take the logarithm (+1 if it has zeros)
    Mstar = np.log10(1.+Mstar)
    Mg = np.log10(1.+Mg)
    Mbh = np.log10(1.+Mbh)
    Mtot = np.log10(Mtot)
    Rstar = np.log10(1.+Rstar)
    Rtot = np.log10(1.+Rtot)
    Rvmax = np.log10(1.+Rvmax)
    GMetal = np.log10(1.+GMetal)
    SMetal = np.log10(1.+SMetal)
    Vmax = np.log10(Vmax)
    Vdisp = np.log10(Vdisp)
    #SFR = np.log10(1.+SFR)

    #normalize the variables
    #Vel_subhalo = normalize(Vel_subhalo)
    Vel_halo = normalize(Vel_halo)

    Mstar = normalize(Mstar)
    Mbh = normalize(Mbh)
    Mg = normalize(Mg)
    Mtot = normalize(Mtot)

    Rstar = normalize(Rstar)
    Rtot = normalize(Rtot)
    Rvmax = normalize(Rvmax)
    GMetal = normalize(GMetal)
    SMetal = normalize(SMetal)
    Vmax = normalize(Vmax)
    Vdisp = normalize(Vdisp)
    V = normalize(V)
    J = normalize(J) 
    #U = normalize(U)
    #K = normalize(K)
    #g = normalize(g)
    SFR = normalize(SFR)

    #1. position
    #2. star mass
    #3. black hole mass 
    #4. gas mass
    #5. total mass 
    #6. Rstar
    #7. Rtot
    #8. Rvmax 
    #9. Gas Metallicity
    #10. Star Metallicity 
    #11. Vmax
    #12. V dispersion 
    #13. V modulo
    #14. J spin modulo
    #15,16,17 photometric bands (remove this)
    #18. SFR
    #tab = np.column_stack((HaloID, Pos_subhalo, Mstar, Mbh, Mg, Mtot, Rstar, Rtot, Rvmax, GMetal, SMetal, Vmax, Vdisp, V, J, SFR))
    tab = np.column_stack((HaloID, Pos_subhalo, Nstars, Mstar, Mbh, Mg, Mtot, Rstar, Rtot, Rvmax, GMetal, SMetal, Vmax, Vdisp, V, J, SFR))
    
    # if you want condition in the number of stars uncomment these and comment the "tab" above
    #tab = tab[tab[:,4]>Nstars_th] # restrict to subhalos with stars
    # Once restricted to a minimum number of stellar particles, remove this feature since it is not observable
    #tab = np.delete(tab, 4, 1)


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
        #if n_sub > 4:  if you want to run the condition of more than 4 subhalos uncomment this and comment the condition above   
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            tab_features = tab_halo[:,3:]
            
            #tab_halo[:,0:3] -= GroupPos[ind]  #in the halo frame
            #tab_halo[:,-3:] -= GroupVel[ind]  
            
            #distance = euclidean_distance(n_sub, torch.Tensor(tab_halo[:,0:3])) 
            #index_mask = (distance > 0) & (distance < r_link)
            #index_edge = np.array(np.where(index_mask == True))
            #index_edge = torch.tensor(index_edge, dtype=torch.long)
            
            #edge_attr = torch.zeros((index_edge.shape[1], 1)) #shape=[number of edges, features=0]
            # TODO: double check if use_loops 
            index_edge, edge_attr = get_edges(tab_halo[:,0:3], r_link, use_loops=True)
            
            u_number = np.log10(n_sub).reshape(1,1) #number of subhalos in the simulation as a global feature
            #print(np.shape(u_number))           

            if global_parameters == True:
                u_parameters = parameters.reshape(1,5)
                #print(np.shape(u_parameters))
                u = np.concatenate((u_number, u_parameters), axis=1)
            elif cosmo_parameters == True:   #esto hay que cambiar
                u_parameters = parameters[0:1]
                u_parameters = u_parameters.reshape(1,1)
                u = np.concatenate((u_number, u_parameters), axis=1)
                #u = u_parameters
                #u = np.concatenate((u_number, u_parameters), axis=1)
            elif astro_parameters == True:
                u_parameters = parameters[2:]
                u_parameters = u_parameters.reshape(1,3) #me parece que aca falta un parametro
                u = np.concatenate((u_number, u_parameters), axis=1)
            else:
                u = u_number
                
            mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
            data = Data(x=torch.Tensor(tab_features), u = torch.tensor(u, dtype=torch.float32), edge_index = torch.tensor(index_edge,  dtype=torch.long), edge_attr = torch.tensor(edge_attr, dtype=torch.float32), y=mass)
            data_sim.append(data)
            
    return data_sim

#-------------------------------------------------------------------------------------------------------------------------------

#mass_sim = np.loadtxt('/mnt/home/bwang/MassGraphNet/data/sobol_sequence_WDM_real_values.txt')
mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')

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
        
        tab, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
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





