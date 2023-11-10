import os
import torch

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import numpy as np
import h5py


from torch_geometric.data import Data
import numpy as np
from torch_geometric.loader import DataLoader

#number of star threshold
Nstar_th = 5

#Use cosmological and astrophysical parameters as global features
global_parameters = False
cosmo_parameters = True  #use only cosmological parameters
astro_parameters = False #use only astrophysical parameters

#in this case I'm creating a training dataset and a validation dataset splitting the simulations
simpathroot = '/data/bcostanza/data'
nsim = 1024

#nhalos = 20 #number of haloes per simulation
#put numbers to choose how many simulations are you taking for training, validation and testing
split_train = 720  #0 to 500 simulations for training
split_valid = 870 #500 to 700 simulations for validation
split_test = nsim #700 to 750 simulations for testing

#name of the dataset 
name_train = 'masswdm_train_menos10_new_sigma8.pt'
name_valid = 'masswdm_valid_menos10_new_sigma8.pt'
name_test = 'masswdm_test_menos10_new_sigma8.pt'

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

#new version of features taken from Cosmo1gal
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
    tab = np.column_stack((HaloID, Pos_subhalo, Mstar, Mbh, Mg, Mtot, Rstar, Rtot, Rvmax, GMetal, SMetal, Vmax, Vdisp, V, J, SFR))
    #tab_features = np.column_stack((Mstar,Rstar,Mdm,Metal))
    #x = torch.tensor(tab_features, dtype=torch.float32)

    
    return tab, Pos_halo, Vel_halo, indexes

#--------------------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------
 
def normalize(variable):
    mean, std = variable.mean(axis=0), variable.std(axis=0)
    norm_variable = (variable - mean)/std
    return norm_variable
    

def create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters):
    
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
            index_mask = (distance > 0) & (distance < 1e-1)
            index_edge = np.array(np.where(index_mask == True))
            index_edge = torch.tensor(index_edge, dtype=torch.long)

            #luego de usar la distance para establecer el rlink quito las posiciones
            tab_features = tab_halo[:,3:]
            
            edge_attr = torch.zeros((index_edge.shape[1], 1)) #shape=[number of edges, features=0]
            
            u_number = np.log10(n_sub).reshape(1,1) #number of subhalos in the simulation as a global feature

            if global_parameters == True:
                u_parameters = parameters.reshape(1,5)
                u = np.concatenate((u_number, u_parameters), axis=1)
            elif cosmo_parameters == True:
                u_parameters = parameters[1:2]
                u_parameters = u_parameters.reshape(1,1)
                u = np.concatenate((u_number, u_parameters), axis=1)
            elif astro_parameters == True:
                u_parameters = parameters[2:5]
                u_parameters = u_parameters.reshape(1,3) #me parece que aca falta un parametro
                u = np.concatenate((u_number, u_parameters), axis=1)
            else:
                u = u_number
                
            mass = torch.tensor(mwdm, dtype=torch.float32) #target
            
            data = Data(x=torch.Tensor(tab_features), u = torch.tensor(u, dtype=torch.float32), edge_index = index_edge, edge_attr = edge_attr, y=mass)
            data_sim.append(data)
            
    return data_sim

#-------------------------------------------------------------------------------------------------------------------------------

mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')

#read the data


def training_set():
    dataset_train = []
    for i in range(0,split_train):
        print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]        
        parameters = mass_sim[i,:-1]  #other parameters of the simulation
    
        halolist = indexes#[:nhalos]
    
        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters)
    
        dataset_train += data_sim
    return dataset_train
    
def validation_set():
    dataset_valid = []
    for i in range(split_train,split_valid):
        print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)
        
        tab, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]
        parameters = mass_sim[i,:-1]
        
        halolist = indexes#[:nhalos]

        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters)

        dataset_valid += data_sim
    return dataset_valid

def test_set():
    dataset_test = []
    for i in range(split_valid,split_test):
        print('reading simulation', i)
        fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)

        tab, GroupPos, GroupVel, indexes = features_new_v2(fin)
        
        mwdm = mass_sim[i,-1]
        parameters = mass_sim[i,:-1]
        
        halolist = indexes#[:nhalos]

        data_sim = create_graphs_new(halolist, tab, GroupPos, GroupVel, mwdm, parameters)

        dataset_test += data_sim

    return dataset_test


print('reading')

train_dataset = training_set()
torch.save(train_dataset, name_train)

valid_dataset = validation_set()
torch.save(valid_dataset, name_valid)
    
test_dataset = test_set()
torch.save(test_dataset, name_test)





