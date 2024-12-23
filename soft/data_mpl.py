import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import optuna
import sys
import math
import h5py


Mbh_std = 1.675002
Mbh_mean = 0.483238518

Mtot_std = 0.57
Mtot_mean = 9.68


simpathroot = '/data/bcostanza/data'
mass_sim = np.loadtxt('/home/bcostanza/MachineLearning/project/sobol_sequence_WDM_real_values.txt')


def normalize_params(params):

    #minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    #maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    minimum = 0.1  #en el caso de omega es 0.1 y 0.5
    maximum = 0.5
    params = (params - minimum)/(maximum - minimum)
    return params


#Mtots = []
data = []

for i in range(0,1024):
    # print('reading simulation', i, flush = True)
    fin = '%s/WDM_%d/fof_subhalo_tab_090.hdf5'%(simpathroot,i)

    f = h5py.File(fin, 'r')

    header = f['Header']
    boxsize = header.attrs[u'BoxSize']

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

    f.close()

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

    #I DON'T DO THIS
    
    #Mbh = (Mbh - Mbh_mean)/Mbh_std
    #Mtot_norm = (Mtot - Mtot_mean)/Mtot_std
    #Mstar = (Mstar - Mstar_mean)/Mstar_std
    #Rstar = (Rstar - Rstar_mean)/Rstar_std
    #Rtot = (Rtot - Rtot_mean)/Rtot_std
    #Rvmax = (Rvmax - Rvmax_mean)/Rvmax_std
    #GMetal = (GMetal - GMetal_mean)/GMetal_std
    #SMetal = (SMetal - SMetal_mean)/SMetal_std
    #Vmax = (Vmax - Vmax_mean)/Vmax_std
    #Vdisp = (Vdisp - Vdisp_mean)/Vdisp_std
    #SFR = (SFR - SFR_mean)/SFR_std
    #J = (J - J_mean)/J_std
    #V = (V - V_mean)/V_std

    mwdm = mass_sim[i, -1]
    
    OmegaM = mass_sim[i, 0]

    #norm omega
    #OmegaM_norm = normalize_params(OmegaM)
    
    #Mtots.append(Mtot)
    data.append(np.hstack([Mtot.mean(), Mtot.std(), OmegaM, mwdm]))
    #data.append(np.hstack([Mstar.mean(), Mstar.std(), Mg.mean(), Mg.std(), Mbh.mean(), Mbh.std(), Mtot.mean(), Mtot.std(), Rstar.mean(), Rstar.std(), Rvmax.mean(), Rvmax.std(), Rtot.mean(), Rtot.std(), GMetal.mean(), GMetal.std(), SMetal.mean(), SMetal.std(), Vmax.mean(), Vmax.std(), Vdisp.mean(), Vdisp.std(), SFR.mean(), SFR.std(), V.mean(), V.std(), J.mean(), J.std(), OmegaM, mwdm]))




data = np.array(data)
print('shape data', np.shape(data))


np.save('mtot_mean_std_norm.npy', data)
