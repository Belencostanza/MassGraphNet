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


device = ""

if torch.cuda.is_available():
    print("CUDA Available")
    #device = torch.device('cuda:'+ sys.argv[1])
    device = torch.device('cuda:1')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


#n_CPU = 5

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


only_mbh = True
only_mtot = False

nfeat = 1  #change to 14 if you use all the features
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
    Rvmax = np.log10(Rvmax)
    GMetal = np.log10(1.+GMetal)
    SMetal = np.log10(1.+SMetal)
    Vmax = np.log10(Vmax)
    Vdisp = np.log10(Vdisp)
    SFR = np.log10(1.+SFR)
    V = np.log10(V)
    J = np.log10(J)


    #obtain mean and std per simulation
    mean_Mg,std_Mg,_ = normalize(Mg)
    mean_Mtot,std_Mtot,_ = normalize(Mtot)
    mean_Mbh,std_Mbh,_ = normalize(Mbh)
    mean_Mstar,std_Mstar,_ = normalize(Mstar)
    mean_Rstar,std_Rstar,_ = normalize(Rstar)
    mean_Rtot,std_Rtot,_ = normalize(Rtot)
    mean_Rvmax,std_Rvmax,_ = normalize(Rvmax)
    mean_GMetal,std_GMetal,_ = normalize(GMetal)
    mean_SMetal,std_SMetal,_ = normalize(SMetal)
    mean_Vmax,std_Vmax,_ = normalize(Vmax)
    mean_Vdisp,std_Vdisp,_ = normalize(Vdisp)
    mean_V,std_V,_ = normalize(V)
    mean_J,std_J,_ = normalize(J) 
    mean_SFR,std_SFR,_ = normalize(SFR)



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


    if only_mbh == True: 
        tab_mean = mean_Mbh
        tab_std = std_Mbh
    elif only_mtot == True: 
        tab_mean = mean_Mtot
        tab_std = std_Mtot
    else: 
        tab_mean = np.stack((mean_Mstar, mean_Mg, mean_Mbh, mean_Mtot, mean_Rstar, mean_Rtot, mean_Rvmax, mean_GMetal, mean_SMetal, mean_Vmax, mean_Vdisp, mean_V, mean_J, mean_SFR))
        tab_std = np.stack((std_Mstar, std_Mg, std_Mbh, std_Mtot, std_Rstar, std_Rtot, std_Rvmax, std_GMetal, std_SMetal, std_Vmax, std_Vdisp, std_V, std_J, std_SFR))

    tab = np.column_stack((HaloID, Pos_subhalo, Mstar, Mg, Mbh, Mtot, Rstar, Rtot, Rvmax, GMetal, SMetal, Vmax, Vdisp, V, J, SFR))

    
#    print(np.shape(tab))
    # if you want condition in the number of stars uncomment these and comment the "tab" above
    # tab = tab[tab[:,4]>Nstars_th] # restrict to subhalos with stars
    # Once restricted to a minimum number of stellar particles, remove this feature since it is not observable
    # tab = np.delete(tab, 4, 1)

    return tab, tab_mean, tab_std, Pos_halo, Vel_halo, indexes#, mean_feat, std_feat
 
def normalize(variable):
    mean, std = variable.mean(axis=0), variable.std(axis=0)
    norm_variable = (variable - mean)/std
    return mean, std, norm_variable

def normalize_params(params):

    #minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    #maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    minimum = 0.1  #en el caso de omega es 0.1 y 0.5
    maximum = 0.5
    params = (params - minimum)/(maximum - minimum)
    return params

def variables(halolist, tab):
    all_features = []
    for ind in halolist:
        n_sub = len(tab[tab[:,0]==ind])
        #num_sub.append(n_sub)

        #if n_sub < 10 and n_sub > 4:
        #if n_sub > 4: # if you want to run the condition of more than 4 subhalos uncomment this and comment the condition above
        tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
        tab_features = tab_halo[:,3:] #(nsub, features)
        all_features.append(tab_features)

    return all_features


def create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm,parameters, r_link, tab_mean, tab_std):
    
    #this for goes through every halo inside the simulation
    #num_sub = []
    data_sim = [] #data in one simulation
    for ind in halolist: 
        n_sub = len(tab[tab[:,0]==ind])
        #num_sub.append(n_sub) 
    
        #if n_sub < 10 and n_sub > 4:
        if n_sub > 4: # if you want to run the condition of more than 4 subhalos uncomment this and comment the condition above   
            tab_halo = tab[tab[:,0]==ind][:,1:]  #select subhalos within a halo with index id (graph por halo)
            tab_features = tab_halo[:,3:]
            
            #mass of that particular Halo in the simulation
            #Halomass = HaloMass[ind].reshape(1,1) 
            #tab_halo[:,-3:] -= GroupVel[ind]  
                        
            index_edge, edge_attr = get_edges(tab_halo[:,0:3], r_link, use_loops=True)
            
            u_number = np.log10(n_sub).reshape(1,1) #number of subhalos in the simulation as a global feature
            #c = np.array([item for pair in zip(mean_feat, std_feat) for item in pair])

            if global_parameters == True:
                u_parameters = parameters.reshape(1,5)
                u = np.concatenate((u_number, u_parameters), axis=1)
            elif cosmo_parameters == True:   #esto hay que cambiar
                u_parameters = parameters[0:1]
                #u_parameters_norm = normalize_params(u_parameters)
                u_parameters = u_parameters.reshape(1,1)
                u_mean = tab_mean.reshape(1,nfeat)
                u_std = tab_std.reshape(1,nfeat)
                #u_c = c.reshape(1,2)
                #u = np.concatenate((u_number, u_parameters, u_c), axis=1)
                #u = np.concatenate((u_number, u_parameters_norm), axis=1)  #without mean and std
                u = np.concatenate((u_number, u_parameters, u_mean, u_std), axis=1)
            elif astro_parameters == True:
                u_parameters = parameters[2:]
                u_parameters = u_parameters.reshape(1,3) #me parece que aca falta un parametro
                u = np.concatenate((u_number, u_parameters), axis=1)
            else:
                u = u_number

                
            mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
            data = Data(x=torch.Tensor(tab_features), u = torch.tensor(u, dtype=torch.float32), edge_index = torch.tensor(index_edge,  dtype=torch.long), edge_attr = torch.tensor(edge_attr, dtype=torch.float32), y=mass, note = tab_features.shape[0])

            data_sim.append(data)
            
    return data_sim

#-------------------------------------------------------------------------------------------------------------------------------

#mass_sim = np.loadtxt('/mnt/home/bwang/MassGraphNet/data/sobol_sequence_WDM_real_values.txt')
mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')

#read the data
def renorm():

    features_all = []
    for i in range(0,split_test):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        tab, GroupPos, GroupVel, indexes = features_new_v2(fin)
        variables_all = tab 
        features_all.append(variables_all)
        print(np.shape(variables_all))
        #variables_all = variables(indexes, tab)

        #features_all += variables_all

    features_all = np.array(np.concatenate(features_all, axis=0))
    print(np.shape(features_all))
    mean = np.mean(features_all, axis=0)
    print(np.shape(mean))
    std = np.std(features_all, axis=0)
    return mean, std

def training_set(r_link):
    dataset_train = []

    for i in range(0,split_train):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, tab_mean, tab_std, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]        
        parameters = mass_sim[i,:-1]  #other parameters of the simulation
    
        halolist = indexes#[:nhalos]
    
        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters, r_link, tab_mean, tab_std)
    
        dataset_train += data_sim
    return dataset_train
    
def validation_set(r_link):

    dataset_valid = []

    for i in range(split_train,split_valid):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, tab_mean, tab_std, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]
        parameters = mass_sim[i,:-1]
        
        halolist = indexes#[:nhalos]

        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters, r_link, tab_mean, tab_std)

        dataset_valid += data_sim
    return dataset_valid

def test_set(r_link):
    dataset_test = []

    #return features_all
    for i in range(split_valid,split_test):
        #print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)

        tab, tab_mean, tab_std, GroupPos, GroupVel, indexes = features_new_v2(fin)
        #print('mean feat:', mean_feat)
        
        mwdm = mass_sim[i,-1]
        parameters = mass_sim[i,:-1]
        
        halolist = indexes#[:nhalos]

        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters, r_link, tab_mean, tab_std)

        dataset_test += data_sim

    return dataset_test

if __name__ == "__main__":
    # dataset_test = test_set(r_link=1e-1)
    # torch.save(dataset_test, 'test_set_all.pt')
    mean_all = []
    std_all = []
   # mean=0.4832
   # std=1.6750
    for i in range(0, split_test):
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        tab_mean, tab_std = features_new_v2(fin)
        mean_all.append(tab_mean)
        std_all.append(tab_std)
    #mean_all, std_all = renorm()
    torch.save(mean_all, 'all_mean_allsim.pt')
    torch.save(std_all, 'all_std_allsim.pt')




