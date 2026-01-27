
import csv
import os
import pynbody
import tangos
import numpy as np
from numpy import sqrt

import gc

from os import listdir
from os.path import *
import sys

import numpy as np 
import pandas as pd 

from darklight import DarkLight

import tangos 

from tangos.examples.mergers import * 

from numpy import sqrt
import random
import pynbody
from .utils import *
from ...config import config

def rank_order_particles_by_angmom(particles):
    
    '''
    Inputs: 

    DMOparticles - Particle data (angular momenta and positions) 
    hDMO - Tangos halo object for the main halo
    
    
    Returns: 
    
    a list of particle IDs ordered by their corresponding angular momenta.
    
    '''
    
    print('this is how many DMOparticles were passed',len(particles))
    

    #particles_in_r200 = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= hDMO['r200c']]
    
    softening_length = pynbody.array.SimArray(np.ones(len(particles))*10.0, units='pc', sim=None)
    
    angular_momenta = get_dist(particles['j'])

    #values arranged in ascending order
    sorted_indicies = np.argsort(angular_momenta.flatten())

    particles_ordered_by_angmom = np.asarray(particles['iord'])[sorted_indicies] if sorted_indicies.shape[0] != 0 else np.array([]) 
   
    return np.asarray(particles_ordered_by_angmom)




def assign_stars_to_particles(snapshot_stellar_mass,particles_sorted_by_angmom,tagging_fraction):
    
    '''

    Tags the lowest angular momenta dark matter particles of a halo with stellar mass. 

    Inputs: 
    
    snapshot_stellar_mass - stellar mass to be tagged in given snapshot 
    particles_sorted_by_angmom - list of particle dark matter IDs sorted by their angular momenta. 
    tagging_fraction - defines the size of the free paramter used to perform tagging 
    
    
    Returns: 

    updates_to_arrays = array updates that need to be written to an output file                  
   
    '''

    size_of_tagging_fraction = int(particles_sorted_by_angmom.shape[0]*tagging_fraction)
    
    particles_in_tagging_fraction = particles_sorted_by_angmom[:size_of_tagging_fraction]
    
    #dividing stellar mass evenly over all the particles in the most bound fraction 

    print('assigning stellar mass')
    
    stellar_mass_assigned = float(snapshot_stellar_mass/len(list(particles_in_tagging_fraction))) if len(list(particles_in_tagging_fraction))>0 else 0
    
    #check if particles have been selected before 
    
    
    #if not selected previously, add to array

    
    array_iords = particles_in_tagging_fraction

    #Uncomment this for old behaviour (where the mass per particle is the total mass tagged upto that point)
    #array_masses = np.append(selected_particles[1][idxs_previously_selected],np.repeat(stellar_mass_assigned,how_many_not_previously_selected))
    array_masses = np.repeat(stellar_mass_assigned,len(array_iords)) 
    
    updates_to_arrays = np.array([array_iords,array_masses])
    
    
    return updates_to_arrays
    


def tag(DMOparticles, hDMO, snapshot_stellar_mass,free_param_value = 0.01, previously_tagged_particles = [np.array([]),np.array([])]):

    '''
    
    Given the dark matter particles and the associated tangos halo object, the function performs particle tagging based on angular momentum 

    Inputs:

    DMOparticles - Particle data (angular momenta, positions, IDs) 
    hDMO - Tangos halo object of main halo 
    snapshot_stellar_mass - stellar mass to be tagged in current snapshot 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    previously_tagged_particles - particle IDs of any previously tagged particles 

    Returns: 
    
    updates_to_arrays = array updates that need to be written to an output file 
    
    
    '''
    particles_ordered_by_angmom = rank_order_particles_by_angmom(DMOparticles)

    return assign_stars_to_particles(snapshot_stellar_mass,particles_ordered_by_angmom, free_param_value)
    


def angmom_tag_over_full_sim(DMOsim, halonumber = 1 ,free_param_value = 0.01, pynbody_path  = None, occupation_frac = 'all' ,particle_storage_filename=None, AHF_centers_file=None, mergers = True):

    '''

    Given a tangos simulation, the function performs angular momentum based tagging over the full simulation. 

    Inputs: 

    DMOsim - tangos simulation 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    pynbody_path - path to particle data 
    occupation_frac - One of 'nadler20' , 'all' , 'edge1' or 'edgert' (controls the occupation regime followed by darklight)
    mergers - Whether to include merging/accreting halos or not. 
    
    Returns: 
    
    dataframe with tagged particle masses at given times, redshifts and associated particle IDs  
    
    '''
    
# path to particle data 
    DMOname = DMOsim.path
    
    # Use config path if pynbody_path not provided
    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")
    # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
    # load_tangos_data is a part of the 'utils.py' file in the tagging dir, it loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
    
    main_halo = DMOsim.timesteps[-1].halos[int(halonumber) - 1]

    # halonums for all snapshots 
    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]

    # time and redshift of each snapshot 
    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
    
    outputs_all = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])
    times_tangos = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])
    
    outputs = outputs_all[np.isin(times_tangos, t_all)]

    outputs.sort()

    # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total =DarkLight(main_halo,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',binning='3bins',timesteps='sim',mergers=True,DMO=True,occupation=2.5e7,fn_vmax=None)

    #calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
    
    if ( len(red_all) != len(outputs) ) : 

        print('output array length does not match redshift and time arrays')
    
    # group_mergers groups all merging objects by redshift.
    # this array gets stored in hmerge_added in the form => len = no. of unique zmerges, 
    # elements = all the hmerges of halos merging at each zmerge
    
    hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)
    
    mstars_total_darklight_l = [] 
    
    # number of stars left over after selection (per iteration)
    leftover=0

    # total stellar mass selected 
    mstar_selected_total = 0

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids = np.array([])

    # if an AHF centering file is provided use the centers stroed within it
    AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_file != None else None

    tagged_iords_to_write = np.array([])
    tagged_types_to_write = np.array([])
    tagged_mstars_to_write = np.array([])
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    
    # looping over all snapshots  
    for i in range(len(outputs)):
        gc.collect()
        
        # was particle data loaded in (insitu) 
        decision=False

        # was particle data loaded in (accreted) 
        decision2=False
        decl = False
    
        print('Current snapshot -->',outputs[i])
    
        # loading in the main halo object at this snapshot from tangos 
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))

        # value of redshift at the current timestep 
        z_val = red_all[i]
                
        # time in gyr
        t_val = t_all[i]

        # 't' is the darklight time array 
        # idrz is thus the index of the mstar value calculated at the closest time to that of the snap
        idrz = np.argmin(abs(t - t_val))

        # index of previous snap's mstar value in darklight array
        idrz_previous = np.argmin(abs(t - t_all[i-1])) if idrz>0 else None 

        # current snap's darklight calculated stellar mass 
        msn = mstar_s_insitu[idrz]              

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            # else msp = previous snap's mstar value
            elif idrz_previous >= 0:
                msp = mstar_s_insitu[idrz_previous]
        else:
            print('There is no stellar mass at current timestep')
            continue

                                                                    
        #calculate the difference in mass between the two mstar's
        mass_select = int(msn-msp)
        print('stellar mass to be tagged in this snap -->',mass_select)

        # if stellar mass is to be tagged then load in particle data 
    
        if mass_select>0:
            
            decision=True
            
            # try to load in the data from this snapshot
            
            try:
                simfn = join(pynbody_path,outputs[i])
                
                print(simfn)
                print('loading in DMO particles')
                
                DMOparticles = pynbody.load(simfn)
                # once the data from the snapshot has been loaded, .physical_units()
                # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                DMOparticles.physical_units()
                
                #print('total energy  ---------------------------------------------------->',DMOparticles['te'])
                print('loaded data insitu')
            
            # where this data isn't available, notify the user.
            except Exception as e:
                print(e)
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
   
            print('mass_select:',mass_select)
            #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
            
            try:
                hDMO['r200c']
            except:
                print("Couldn't load in the R200 at timestep:" , i)
                continue
            
            print('the time is:',t_all[i])
        
            subhalo_iords = np.array([])
            
            if type(AHF_centers_file) == type(None):
                print(int(halonums[i])-1)
                h = DMOparticles.halos()[int(halonums[i])-1]
            
            elif type(AHF_centers_file) != type(None):
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]
                
                h = DMOparticles.halos(halo_numbers="v1")[int(AHF_crossref)] 
                
                # the "children" are subhalos that need to be removed before centering on the main halo
                children_ahf_int = h.properties['children']
            
                halo_catalogue = DMOparticles.halos(halo_numbers="v1")
            
                subhalo_iords = np.array([])
                
                for ch in children_ahf_int:
                    
                    if ch != AHF_crossref: 
                        subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(ch)].dm['iord'])
                                                                                                                                        
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
                

            pynbody.analysis.halo.center(h)

            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        
            try:                                                                                                                                                                                              
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
                                                                                                                                                                                                              
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
                                                                                                                                                                            
                        
            DMOparticles_insitu_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]

            #print('angular_momentum: ', DMOparticles["j"])
            
            DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]

            #DMOparticles_insitu_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],accreted_only_particle_ids))]

            particles_sorted_by_angmom = rank_order_particles_by_angmom( DMOparticles_insitu_only)
            
            if particles_sorted_by_angmom.shape[0] == 0:
                continue
            
            array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(free_param_value))
            
            print('writing insitu particles to output file')
            
            tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write[0])
            tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('insitu',len(array_to_write[0])))
            tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write[1])
            ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write[0])))
            zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write[0])))

            
            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
            
            #pynbody.analysis.halo.center(h,mode='hyb').revert()

            del DMOparticles_insitu_only
            
            #print('moving onto mergers loop')
            #get mergers ----------------------------------------------------------------------------------------------------------------
            # check whether current the snapshot has a the redshift just before the merger occurs.
        
        if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):
                
            decision2 = False if decision==True else True

            decl=False
            
            t_id = int(np.where(z_set_vals==red_all[i+1])[0][0])

            #print('chosen merger particles ----------------------------------------------',len(chosen_merger_particles))
            #loop over the merging halos and collect particles from each of them
        
            #mstars_total_darklight = np.array([])
            DMO_particles = 0 
            
            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:',hDM)
                
                if (occupation_frac != 'all'):
                    try:
                        prob_occupied = calculate_poccupied(hDM,2.5e7)

                    except Exception as e:
                        print(e)
                        print("poccupied couldn't be calculated")
                        continue
                    
                    if (np.random.random() > prob_occupied):
                        print('Skipped')
                        continue
                #angmom_tag_over_full_sim(hDM, free_param_value = 0.01, pynbody_path  = pynbody_path, occupation_frac = 'all', mergers = True)

                
                try:
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',binning='3bins',timesteps='sim',mergers=True,DMO=False,occupation=2.5e7,fn_vmax=None)
                    

                    


                    #occupation='edge1', pre_method='fiducial_with_turnover', post_scatter_method='flat', DMO=True,mergers = True)
                    #DarkLight(hDM,DMO=True)#,poccupied=occupation_frac,mergers=True)
                    print(len(t_2))
                    print(mstar_merging)
                except Exception as e :
                    print(e)
                    print('there are no darklight stars')
                    #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                
                    continue
        
        
                if len(mstar_merging)==0:
                    #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                    continue

                mass_select_merge= mstar_merging[-1]
                #mstars_total_darklight = np.append(mstars_total_darklight,mass_select_merge)

                print(mass_select_merge)
                if int(mass_select_merge)<1:
                    leftover+=mstar_merging[-1]
                    continue
                
                simfn = join(pynbody_path, outputs[i])

                if float(mass_select_merge) >0 and decision2==True:
                    # try to load in the data from this snapshot
                    try:
                        DMOparticles = pynbody.load(simfn)
                        DMOparticles.physical_units()
                    
                        print('loaded data in mergers')
                    # where this data isn't available, notify the user.
                    except:
                        print('--> DMO particle data exists but failed to read it, skipping!')
                        continue
                    decision2 = False
                    decl=True
             
                if int(mass_select_merge) > 0:

                    try:
                        h_merge = DMOparticles.halos()[int(hDM.calculate('halo_number()'))-1]
                        pynbody.analysis.halo.center(h_merge.dm)
                        
                        #r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                    except Exception as ex:
                        print('centering data unavailable, skipping',ex)
                        continue
                                                                                                           
                    r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                    print('mass_select:',mass_select_merge)
                    #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
                    print('sorting accreted particles by TE')
                    #print(rank_order_particles_by_te(z_val, DMOparticles, hDM,'accreted'), 'output')
                    DMOparticles_acc_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn_acc] 

                    #DMOparticles_acc_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],insitu_only_particle_ids))]
                                            
                    try:
                        accreted_particles_sorted_by_angmom = rank_order_particles_by_angmom(DMOparticles_acc_only)
                    except:
                        continue
                    
        
                    print('assinging stars to accreted particles')

                    array_to_write_accreted = assign_stars_to_particles(mass_select_merge,accreted_particles_sorted_by_angmom,float(free_param_value))
                    
                    
                    tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write_accreted[0])
                    tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('accreted',len(array_to_write_accreted[0])))
                    tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write_accreted[1])
                    ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write_accreted[0])))
                    zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write_accreted[0])))

        
                    accreted_only_particle_ids = np.append(accreted_only_particle_ids,np.asarray(array_to_write_accreted[0]))
                    print('writing accreted particles to output file')
          
                    del DMOparticles_acc_only
        
                  
                            
        if decision==True or decl==True:
            del DMOparticles
    
    
        print("Done with iteration",i)

        df_tagged_particles = pd.DataFrame({'iords':tagged_iords_to_write, 'mstar':tagged_mstars_to_write,'t':ts_to_write,'z':zs_to_write,'type':tagged_types_to_write})

        if particle_storage_filename != None:
            df_tagged_particles.to_csv(particle_storage_filename)
            
    return df_tagged_particles



def integrate_sfr(sfr,tend):

    #print("length of sfr:",len(sfr))                                                                                                                  

    mstar_per_gyr = sfr*(10**9)

    #print("length of sfr:",len(mstar_per_gyr))                                                                                                        

    mstar = mstar_per_gyr*0.02

    #print("length of:",len(mstar))                                                                                                                    
    time_bins = np.arange(0,tend,0.02)

    t_array = time_bins[1:]

    mstar_array = np.cumsum(mstar)
    #print("length of:",len(mstar_array))                                                                                                              
    return mstar_array,t_array






def angmom_tag_over_full_sim_recursive(DMOsim,tstep, halonumber, free_param_value = 0.001,free_param_value_acc = None ,pynbody_path  = None, particle_storage_filename=None, AHF_centers_filepath=None, mergers = True, df_tagged_particles=None ,tag_typ='insitu',acc_halo_path_tagged=None,main_halo_paths=None,mode="darklight"):

    '''

    Given a tangos simulation, the function performs angular momentum based tagging over all its snapshots.
    Recursively tags accreting halos down the merger tree over their entire lifetimes 

    Inputs: 

    DMOsim - tangos simulation 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    pynbody_path - path to particle data 
    occupation_frac - One of 'nadler20' , 'all' , 'edge1' or 'edgert' (controls the occupation regime followed by darklight)
    mergers - Whether to include merging/accreting halos or not. 
    
    Returns: 
    
    dataframe with tagged particle masses at given times, redshifts and associated particle IDs  
    
    '''

    if type(free_param_value_acc) == type(None): 
        free_param_value_acc = free_param_value

    #sets halo catalogue priority to HOP by default  (because all the EDGE tangos db are currently hop based)
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

# extracts name of DMO simulation
    DMOname = DMOsim.path
    
    # Use config path if pynbody_path not provided
    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")
    
    # load-in tangos data upto given timestep
    main_halo = DMOsim.timesteps[tstep].halos[int(halonumber) - 1]

    # halonums for all snapshots 
    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]

    # time and redshift of each snapshot 
    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
    
    outputs_all = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])
    times_tangos = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])
    
    print("outputs:",outputs_all)
    print("times:",t_all)

    # names of simulation output files 
    outputs = outputs_all[np.isin(times_tangos, t_all)]

    outputs.sort()
                                    
    # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False, excludes accreted mass)
    
    if mode == "darklight":
        t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',binning='3bins',timesteps='sim',mergers=False,DMO=False,occupation=2.5e7,fn_vmax=None)

    else: 
        mstar_s_insitu,t = integrate_sfr(main_halo["SFR_histogram"],t_all[-1])
        redshift = main_halo.calculate_for_progenitors('z()')[0][::-1]                                                                                     

    # calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
    # these are based on the HOP catalogue by default 
    
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)

    # check time and output array have same size 
    if ( len(red_all) != len(outputs) ) : 
        print('output array length does not match redshift and time arrays')
    
    # group_mergers groups all merging halo objects by redshift.
    hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

    mstars_total_darklight_l = []
    
    # number of stars left over after selection (per iteration)
    leftover = 0

    # total stellar mass selected 
    mstar_selected_total = 0

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids = np.array([])

    # if an AHF centering file is provided use the centers stroed within it
    AHF_centers = pd.read_csv(os.path.join(AHF_centers_filepath,str(DMOname)+".csv")) if type(AHF_centers_filepath) != type(None) else None
    AHF_centers_acc = pd.read_csv(os.path.join(AHF_centers_filepath,str(DMOname)+"_accreted.csv")) if type(AHF_centers_filepath) != type(None) else None
    
    tagged_iords_to_write = np.array([])
    tagged_types_to_write = np.array([])
    tagged_mstars_to_write = np.array([])
    
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    
    # record of tagged objects for the recursive run where the loop goes through all merging objects 
    acc_halo_path_tagged = np.array([]) if (type(acc_halo_path_tagged) == type(None)) else acc_halo_path_tagged 



    if len(acc_halo_path_tagged) > 0:

        halo_path = main_halo.calculate_for_progenitors('path()')
        print("halopath:",halo_path)
        main_halo_paths = np.array([])
        main_halo_paths = np.append(main_halo_paths,halo_path[0][0])

        if ( len(np.where(np.isin(main_halo_paths,acc_halo_path_tagged) == True)[0]) > 0):

            print("overlap at : ",main_halo_paths[np.where(np.isin(main_halo_paths,acc_halo_path_tagged) == True)])
            print("for halo : ",acc_halo_path_tagged )
            return df_tagged_particles,acc_halo_path_tagged


    halo_path = main_halo.calculate_for_progenitors('path()')
    acc_halo_path_tagged = np.append(acc_halo_path_tagged,halo_path[0][0])



    if  type(df_tagged_particles) == type(None):    
        df_tagged_particles = pd.DataFrame({'iords':tagged_iords_to_write, 'mstar':tagged_mstars_to_write,'t':ts_to_write,'z':zs_to_write,'type':tagged_types_to_write})
    
    

    # looping over all snapshots  
    for i in range(len(outputs)):
        gc.collect()
        if len(t) == 0:
            continue
        # was particle data loaded in (insitu) 
        decision=False

        # was particle data loaded in (accreted) 
        decision2=False
        decl = False
    
        print('Current snapshot -->',outputs[i])
    
        # loading in the main halo object at this snapshot from tangos 
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))

        # value of redshift at the current timestep 
        z_val = red_all[i]
                
        # time in gyr
        t_val = t_all[i]

        # 't' is the darklight time array 
        # idrz is thus the index of the mstar value calculated at the closest time to that of the snap
        idrz = np.argmin(abs(t - t_val))

        # index of previous snap's mstar value in darklight array
        idrz_previous = np.argmin(abs(t - t_all[i-1])) if idrz>0 else None 

        # current snap's darklight calculated stellar mass 
        msn = mstar_s_insitu[idrz]              

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap idrz_previous==None and msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            elif idrz_previous >= 0:
                msp = mstar_s_insitu[idrz_previous]
        else:
            print('There is no stellar mass at current timestep')
            continue

                                                                    
        #calculate the difference in mass between the two mstar's
        mass_select = int(msn-msp)
        print('stellar mass to be tagged in this snap -->',mass_select)

        # if stellar mass is to be tagged then load in particle data 
    
        if mass_select>0:
            if type(AHF_centers_filepath) != type(None):
                # if AHF centers are available then the priority is changed to the AHF catalogue (Which is 1 indexed)
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
            
            decision=True
            
            # try to load in the data from this snapshot
            
            try:
                simfn = join(pynbody_path,outputs[i])
                
                print(simfn)
                print('loading in DMO particles')
                
                DMOparticles = pynbody.load(simfn)
                # once the data from the snapshot has been loaded, .physical_units()
                # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                DMOparticles.physical_units()
                #DMOparticles = DMOparticles.d 
                print('loaded data insitu')
            
            # where this data isn't available, notify the user.
            except Exception as e:
                print(e)
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
   
            print('mass_select:',mass_select)
            
            try:
                hDMO['r200c']
            except:
                print("Couldn't load in the R200 at timestep:" , i)
                continue
            
            print('the time is:',t_all[i])
        
            subhalo_iords = np.array([])
            
            if type(AHF_centers_filepath) == type(None):
                print("Halonum:",int(halonums[i])-1)
                
                # if the AHF centers are unavailable, the default HOP catalogue is used (which is zero indexed)
                h = DMOparticles.halos()[int(halonums[i])-1]
                #h = h.dm
            
            elif type(AHF_centers_filepath) != type(None):
                # if AHF centers are available then the priority is changed to the AHF catalogue (Which is 1 indexed)
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]
                
                h = DMOparticles.halos(halo_numbers="v1")[int(AHF_crossref)] 
                #h = h.dm
                # the "children" are subhalos that need to be removed before centering on the main halo
                children_ahf_int = h.properties['children']
            
                halo_catalogue = DMOparticles.halos(halo_numbers="v1")
            
                subhalo_iords = np.array([])
                
                for ch in children_ahf_int:
                    
                    if ch != AHF_crossref: 
                        subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(ch)].dm['iord'])
                                                                                                                                        
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
            

            pynbody.analysis.halo.center(h.dm)
            pynbody.analysis.angmom.faceon(h.dm[h.dm['r']<5])
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
        
            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
            
            DMOparticles_insitu_only = DMOparticles.dm[sqrt(DMOparticles.dm['pos'][:,0]**2 + DMOparticles.dm['pos'][:,1]**2 + DMOparticles.dm['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]
        
            DMOparticles_insitu_only = DMOparticles_insitu_only.dm
            #uncomment to remove subhalos from tagging insitu 

            ####DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]
            
            particles_sorted_by_angmom = rank_order_particles_by_angmom( DMOparticles_insitu_only.dm)
            
            if particles_sorted_by_angmom.shape[0] == 0:
                continue
            
            array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(free_param_value))
            
            print('writing insitu particles to output file')
            
            tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write[0])
            tagged_types_to_write = np.append(tagged_types_to_write,np.repeat(tag_typ,len(array_to_write[0])))
            tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write[1])
            ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write[0])))
            zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write[0])))

            row_to_write = pd.DataFrame({'iords':array_to_write[0], 'mstar':array_to_write[1],'t':np.repeat(t_all[i],len(array_to_write[0])),'z':np.repeat(red_all[i],len(array_to_write[0])) , 'type':np.repeat(tag_typ,len(array_to_write[0])) })

            df_tagged_particles =  pd.concat([df_tagged_particles,row_to_write],ignore_index=True)
            
            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))

            del DMOparticles_insitu_only
            
            #get mergers ----------------------------------------------------------------------------------------------------------------
            # check whether current the snapshot has a the redshift just before the merger occurs.
        
        if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):
                
            decision2 = False if decision==True else True

            decl=False
            
            t_id = int(np.where(z_set_vals==red_all[i+1])[0][0])

            #print('chosen merger particles ----------------------------------------------',len(chosen_merger_particles))
            #loop over the merging halos and collect particles from each of them
    
            DMO_particles = 0 
            
            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:',hDM)
                
                #if (occupation_frac != 'all'):
                try:
                    prob_occupied = calculate_poccupied(hDM,2.5e7)
                    #prob_occupied = 1
                except Exception as e:
                    print(e)
                    print("poccupied couldn't be calculated")
                    continue
                    
                if (np.random.random() > prob_occupied):
                    print('Skipped')
                    continue
                try:
                    if mode == 'darklight':
                        t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,nscatter=0,vthres=26.3,zre=4.,pre_method='fiducial',post_method='schechter',post_scatter_method='increasing',binning='3bins',timesteps='sim',mergers=True,DMO=False,occupation=2.5e7,fn_vmax=None)
                    
                    else:
                        t_2= hDM.calculate_for_progenitors("t()")[0][::-1]                                                                                 
                        redshift_2= hDM.calculate_for_progenitors("z()")[0][::-1]                                                                          
                        mstar_in2 = hDM.calculate_for_progenitors("M200c_stars")[0][::-1]                                                                  
                        mstar_merging = mstar_in2                                                                                                         
                        #mstar_in2,t_2 = integrate_sfr(hDM["SFR_histogram"],t_all[-1])
                        
                    #occupation=occupation_frac, pre_method='fiducial_with_turnover', post_scatter_method='flat',DMO=True,mergers = True)
                    #occupation=2.5e7, pre_method='fiducial',post_method='fiducial',post_scatter_method='flat', DMO=True, mergers=True)
                    #occupation=2.5e7, pre_method='fiducial', post_method='fiducial', post_scatter_method='flat'
                except Exception as e :
                    print(e)
                    print('there are no darklight stars')
                    continue

                if len(mstar_merging) == 0:
                    print("Darklight unable to make predictions")
                    continue
                
                if len(np.where(np.asarray(mstar_merging) > 0)[0]) == 0:
                    print("Darklight predicts no stars in this halo")
                    continue
                                                                                                                                    
                tidx = np.where(np.asarray(DMOsim.timesteps[:]) ==  hDMO.timestep)[0][0]
                acc_halo_path = hDM.calculate_for_progenitors('path()')
                print('halonum merging:',hDM.calculate_for_progenitors('halo_number()'))
                halonumber_hDM = hDM.calculate_for_progenitors('halo_number()')[0][0]

                print('halonum merging:',halonumber_hDM)
                

                if type(main_halo_paths) != type(None): 
                    
                    if ( len(np.where(np.isin(main_halo_paths,acc_halo_path_tagged) == True)[0]) != 0):
                        continue

                # if halo has not been tagged on before, we want to perform tagging over its full lifetime (upto the current snap)
                if ( len(np.where(np.isin(acc_halo_path,acc_halo_path_tagged) == True)[0]) == 0 ):
                    
                    ######################change
                    print('---recursion triggered -----')
                    df_tagged_particles,acc_halo_path_tagged = angmom_tag_over_full_sim_recursive(DMOsim,tidx,halonumber_hDM, free_param_value = float(free_param_value_acc),free_param_value_acc = float(free_param_value_acc),pynbody_path = pynbody_path, df_tagged_particles=df_tagged_particles,tag_typ='accreted',acc_halo_path_tagged=acc_halo_path_tagged,main_halo_paths=main_halo_paths)
                    ##################
                    
                    
                    print('---recursion end -----')
                                
                    
                else:
                    #HOP halonum,AHF halonum,snapshot
                    if len(mstar_merging)==0:
                        continue
    
                    mass_select_merge= mstar_merging[-1] - mstar_merging[-2]  if len(mstar_merging) > 1 else mstar_merging[-1]
    
                    print(mass_select_merge)
                    if int(mass_select_merge)<1:
                        
                        continue
                    
                    simfn = join(pynbody_path, outputs[i])
                    
                    #if type(AHF_centers_filepath) != type(None):
                     #   pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]


                    if float(mass_select_merge) >0 and decision2==True:
                        # try to load in the data from this snapshot
                        
                        if type(AHF_centers_filepath) != type(None):
                            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]

                        try:
                            DMOparticles = pynbody.load(simfn)
                            DMOparticles.physical_units()
                            #DMOparticles = DMOparticles.d
                            print('loaded data in mergers')
                        # where this data isn't available, notify the user.
                        except:
                            print('--> DMO particle data exists but failed to read it, skipping!')
                            continue
                        decision2 = False
                        decl=True

                    
                    if int(mass_select_merge) > 0:
    
                        try:

                            HOP_halonum_acc = int(hDM.calculate('halo_number()'))

                            if (type(AHF_centers_filepath) != type(None)):
                                AHF_halonum_acc = AHF_centers_acc[AHF_centers_acc["snapshot"] == outputs[i]]
                                AHF_halonum_accreted = AHF_halonum_acc[AHF_halonum_acc["HOP halonum"] == HOP_halonum_acc]["AHF halonum"].values[0]
                                h_merge = DMOparticles.halos(halo_numbers="v1")[AHF_halonum_accreted]

                            else:

                                h_merge =  DMOparticles.halos()[HOP_halonum_acc - 1]

                            pynbody.analysis.halo.center(h_merge,mode='hyb')

                            r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')

                        except Exception as ex:
                            print('centering data unavailable, skipping',ex)
                            continue

                        print('mass_select:',mass_select_merge)
                        #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
                        print('sorting accreted particles by Angmom.')
                        #print(rank_order_particles_by_te(z_val, DMOparticles, hDM,'accreted'), 'output')
        
                        DMOparticles_acc_only = DMOparticles.d[sqrt(DMOparticles.d['pos'][:,0]**2 + DMOparticles.d['pos'][:,1]**2 + DMOparticles.d['pos'][:,2]**2) <= r200c_pyn_acc] 
                                                    
                        try:
                            accreted_particles_sorted_by_angmom = rank_order_particles_by_angmom(DMOparticles_acc_only.dm)
                        except:
                            continue
                        
            
                        print('assinging stars to accreted particles')
    
                        array_to_write_accreted = assign_stars_to_particles(mass_select_merge,accreted_particles_sorted_by_angmom,float(free_param_value_acc))
                        
    
                        tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write_accreted[0])
                        tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('accreted',len(array_to_write_accreted[0])))
                        tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write_accreted[1])
                        ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write_accreted[0])))
                        zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write_accreted[0])))
    
            
                        accreted_only_particle_ids = np.append(accreted_only_particle_ids,np.asarray(array_to_write_accreted[0]))
                        row_to_write_acc = pd.DataFrame({'iords':array_to_write_accreted[0], 'mstar':array_to_write_accreted[1],'t':np.repeat(t_all[i],len(array_to_write_accreted[0])),'z':np.repeat(red_all[i],len(array_to_write_accreted[0])) , 'type':np.repeat('accreted',len(array_to_write_accreted[0])) })
                        
                        df_tagged_particles = pd.concat([df_tagged_particles,row_to_write_acc],ignore_index=True)            

                        print('writing accreted particles to output file')
              
                        del DMOparticles_acc_only
                  
                            
        if decision==True or decl==True:
            del DMOparticles
    
    
        print("Done with iteration",i)


        if particle_storage_filename != None:
            df_tagged_particles.to_csv(particle_storage_filename)
            
    return df_tagged_particles,acc_halo_path_tagged




    
    
    
    
    
