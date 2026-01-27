import numpy as np 
import pandas as pd 
import darklight  
from numpy import sqrt
import random
import pynbody

def initialize_arrays(n):
    x = []
    for i in range(n):
        x.append(np.array([]))
        
    return x

def get_dist(pos):

    # calculates the magnitude of the 3D position vector given 
    # a 3D 'pos' array 

    return np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
                    


def load_indexing_data(DMOsim,halo_number):
    
    main_halo = DMOsim.timesteps[-1].halos[int(halo_number) - 1]
    
    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
   
    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all = main_halo.calculate_for_progenitors('z()')[0][::-1] 
    
    outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])
    times_tangos = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

    valid_outputs = outputs[np.isin(times_tangos, t_all)]
    
    valid_outputs.sort()
    #snapshots = [ f for f in listdir(pynbody_path+DMOname) if (isdir(join(pynbody_path,DMOname,f)) and f[:6]=='output') ]
    return t_all, red_all, main_halo, halonums, valid_outputs




def calculate_poccupied(halo_object,occupation_regime):

    # units = kpc^3 M_sun^-1 s^-2
    G_constant = 4.3009*(10**(-6))

    # units = kpc s^-1 
    vmax = max(np.sqrt( G_constant * (halo_object['dm_mass_profile']/halo_object['rbins_profile']) ))

    m200 = halo_object['M200c']
    p_occupied = darklight.core.occupation_fraction(vmax,m200,method=occupation_regime)

    return p_occupied

    
def group_mergers(z_merges,h_merges):

    #groups the halo objects of merging halos by redshift
    
    merging_halos_grouped_by_z = []
    
    #redshifts at which mergers took place
    z_unique_values = sorted(list(set(z_merges)))
    
    # for each of these redshifts
    for i in z_unique_values:
        # indices of halo objects merging at given redshift 'i'
        # zmerges = 1D (can contain 'i' multiple times)
        lists_of_halos_merging_at_current_z = np.where(z_merges==i)
        
        all_halos_merging_at_current_z=[]
        
        # collect halo objects of merging halos for the redhsift 'i'
        for list_of_halos in lists_of_halos_merging_at_current_z :
            halos_merging_at_current_z =np.array([])
            
            
            for merging_halo_object in list_of_halos:
                halos_merging_at_current_z = np.append(halos_merging_at_current_z, h_merges[merging_halo_object][1:])

            all_halos_merging_at_current_z.append(halos_merging_at_current_z)
        
        merging_halos_grouped_by_z.append(all_halos_merging_at_current_z)
    
    return merging_halos_grouped_by_z, z_unique_values

