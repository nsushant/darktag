# parent = pynbody_analysis.py created 2021.08.11 by Stacy Kim

# selection script = created 2021.08.21 by Sushanta Nigudkar 


#import tracemalloc
#from memory_profiler import profile


import csv
import os
import pynbody
import tangos
import numpy as np
from numpy import sqrt
from darklight import DarkLight
import darklight 
from os import listdir
from os.path import *
import gc
from tangos.examples.mergers import *     
import random
import sys
import pandas as pd
from darktrace.tagging.spatial_tagging import *
from darktrace.tagging.utils import *
from config import config


def help():

    print('tag_particles(name of simulation, occupation fraction for darklight,filename of particle_data file to be created)')
    print('calculate_reffs(sim_name, particles_tagged,reffs_fname,from_file = (True if particles_tagged are to be read from file),from_dataframe= True if particles_tagged are to be read from data frame, save_to_file=True)')

    
    return



def spatial_tag_particles(sim_name_input,occupation_fraction,filename_for_run,mergers=True):
    
#used paths
#used paths
# set the correct paths to data files
    tangos_path_edge     = config.get_path("tangos_path")
    tangos_path_chimera  = config.get_path("tangos_path")
    pynbody_path_edge    = config.get_path("pynbody_path")
    pynbody_path_chimera = config.get_path("pynbody_path")
    pynbody_edge_gm = config.get_path("pynbody_path")


    # you can comment out any that you don't want to run here
    '''
    sims = [#'Halo383_fiducial'
             #'Halo383_fiducial_late',   'Halo383_fiducial_288', 'Halo383_fiducial_early' 'Halo383_Massive'
           #'Halo600_fiducial',
        #'Halo600_fiducial_later_mergers',
        #'Halo1445_fiducial'
            #'Halo605_fiducial'#,
            #'Halo624_fiducial'#, 'Halo624_fiducial_higher_finalmass','Halo1459_fiducial',
        #'Halo605_fiducial'#,
     #'Halo1459_fiducial_Mreionx02'#, 'Halo1459_fiducial_Mreionx03', 'Halo1459_fiducial_Mreionx12','Halo600_RT', 'Halo605_RT', 'Halo624_RT',
        #'Halo1445_RT' #, 'Halo1459_RT'
    ]
    '''
    sims = [sim_name_input]
    
    # keeps count of the number of mergers
    mergers_count = 0
        
    # iterating over all the simulations in the 'sims' list
    for isim,simname in enumerate(sims):

        print('==================================================')
        print(simname)

        # assign it a short name
        split = simname.split('_')
        shortname = split[0][4:]
        halonum = shortname[:]
        if len(split) > 2:
            if   halonum=='332': shortname += 'low'
            elif halonum=='383': shortname += 'late'
            elif halonum=='600': shortname += 'lm'
            elif halonum=='624': shortname += 'hm'
            elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
            elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
            elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
            else:
                print('unsupported simulation',simname,'! Not sure what shortname to give it. Aborting...')
                continue
        elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'

        if simname[-3] == 'x':
            DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

        else:
            DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))
                                
        #DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:]))) #if split[1]=='fiducial' else None
        
        
        # set the correct paths to data files
        if halonum == '383':
            tangos_path  = tangos_path_chimera
            pynbody_path = pynbody_path_chimera
        else:
            tangos_path  = tangos_path_edge
            pynbody_path = pynbody_path_edge if halonum == shortname else pynbody_edge_gm
        
        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO particle does not data exists, skipping!')
            continue
        
        # listdir returns the list of entries in a given dir path (like ls on a dir)
        # isdir check if the given dir exists
        # join creates a string consisting of the path,name,entry in dir
        # once we have this string we check to see if the word 'output' is in this string (to grab only the output snapshots)
                    
        snapshots = [ f for f in listdir(pynbody_path+DMOname) if (isdir(join(pynbody_path,DMOname,f)) and f[:6]=='output') ]

        #sort snapshots array
        snapshots.sort()
        
        # load in the DMO sim to get particle data and get accurate halonums, main halo object, in each snapshot
        DMOsim,main_halo,halonums,outputs = load_indexing_data_and_halo_object(DMOname,1)
            
        #darklight stellar masses used for the selection of insitu particles
        t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers=False,poccupied=occupation_fraction)
        
        #calculate when the mergers took place (zmerge) and grab all the halo objects involved in the merger (hmerge)
        zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
        
        red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
       
        t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
        
        if ( len(red_all) != len(outputs) ) : 

            print('output array length does not match redshift and time arrays')

        # group_mergers() groups all the non-main halo objects that take part in mergers according to the merger redshift.
        # this array gets stored in hmerge_added in the form -> len = no. of unique zmerges (redshifts of mergers),
        # elements = all the halo objects of halos merging at this redshift
        
        hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

        #print the total amount of the (insitu) stellar mass that is to be associated with particles at this snap
        print('dkl',np.array(mstar_s_insitu))

        #initialize '12' empty arrrays named as shown (for storage of calculated tagging parameters and particle IDs)
        part_typ,time_of_choice,redshift_of_choice,chosen_parts,pos_choice_x,pos_choice_y,pos_choice_z,m_tot,a_coeff,a_coeff_merger,a_coeff_tot,output_number = initialize_arrays(12)

        tagged_stellar_masses = []
        
        # number of stars left over after selection (per iteration)
        leftover=0

        # total stellar mass selected
        mstar_selected_total = 0

        # looping over all snapshots
        for i in range(len(outputs)):
        
            # take a tally of all the particles chosen before this snap
            print('This is how many particles have been chosen:',len(chosen_parts))
            
            # was particle data loaded in the insitu cycle of tagging ?
            decision=False
        
            # was particle data loaded in the accreted cycle of tagging ?
            decision2=False
            decl = False
            
            # Garbage collection
            gc.collect()
            
            # lets the user know what snapshot is being currently processed
            print('output',i)
            
            ## confirms that particle have been chosen
            if chosen_parts.shape[0]>0:
                
                #if they have, then chaekc if they are all unique (if not inform the user)
                if chosen_parts.shape[0] != len(set(chosen_parts)):
                    print('unequal arrays!! duplicates present',chosen_parts.shape[0],len(set(chosen_parts)))


          
            # load the main tangos halo object
            hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))
 
            # value of redshift at the current timestep
            z_val = red_all[i]
            t_val = t_all[i]
            
            # round each value in the redhsifts list from DarkLight to 6 decimal places
            np_round_to_6 = np.round(np.array(abs(redshift)), 6)
          
            # generate path to snapshot 
            simfn = join(pynbody_path,DMOname,outputs[i])

            # here t = darklight time array , t_val = time associated with current snapshot 
            idrz = np.argmin(abs(t - t_val))
                
            # index of previous snap's mstar value in darklight array
            idrz_previous = np.argmin(abs(t - t_all[i-1])) if idrz>0 else None 

            msn = mstar_s_insitu[idrz]

            #get the index at which the redshift of the snapshot is stored in the DarkLight array
        
            if msn != 0:
                if idrz_previous==None:
                    msp = 0
                elif idrz_previous >= 0:
                    msp = mstar_s_insitu[idrz_previous]
            else:
                print('There is no stellar mass at current timestep')
                continue
                
            # Using the index obtained in idrz get the value of mstar_insitu at this redshift
                                                                                                                                                                                                          
            print('stellar masses [now,previous] :',msn,msp)

            #calculate the difference in mass between the two mstar's
            mass_select = int(msn-msp) if msn > 0 else 0
            
            if mass_select>1112:
                decision=True
                # try to load in the data from this snapshot
                
                try:
                    DMOparticles = pynbody.load(simfn)
                    # once the data from the snapshot has been loaded, .physical_units()
                    # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                    DMOparticles.physical_units()
                    #print('Mass units ----------------------------------------------------->',DMOparticles['mass'].in_units('1.00e+10 Msol h**-1'))
                    print('loaded data insitu')
                    
                # where this data isn't available, notify the user.
                except:
                    print('--> DMO particle data exists but failed to read it, skipping!')
                    continue
           
                print('mass_select:',mass_select)
                
                #the pynbody halo object of the main halo
                h = DMOparticles.halos()[int(halonums[i])-1]
                
                #center the simulation snapshot on the main halo obtained above
                pynbody.analysis.halo.center(h,mode='hyb')
                   
                r200 = hDMO['r200c']
                
                print(r200, '-------------------------------R200')

                a_check = plum_const(hDMO,z_val,'insitu',r200)

                a_coeff = np.append(a_coeff,a_check)
                # filter out particles outside the plummer tidal radius 'a'.
                particles_within_selection_distance = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= 10*a_check ]

                if len(particles_within_selection_distance)==0:
                    print('no particles in the selection radius')
                    continue
                #chaged M_0!!! 
                binned_df, bins,a,a_coeff,selection_mass = prod_binned_df(z_val, msn,mass_select,chosen_parts,DMOparticles,hDMO,'insitu',a_coeff,r200)
                ## perform the selection for each bin
                choose_parts,output_num,tgyr_of_choice,r_of_choice,p_typ,a_storage,m_storage = get_bins(bins, binned_df, mass_select, a, a_coeff, msp,red_all,t_all,i,'insitu',selection_mass)
                
                if len(choose_parts)>0:
                        
                        chosen_parts = np.append(chosen_parts,choose_parts)
                        output_number = np.append(output_number, output_num)
                        redshift_of_choice = np.append(redshift_of_choice,r_of_choice)
                        
                        time_of_choice = np.append(time_of_choice,tgyr_of_choice)
                        #a_coeff = np.append(a_coeff,a_storage)
                        
                        part_typ = np.append(part_typ, p_typ)
                        a_coeff_tot = np.append(a_coeff_tot, a_storage)
                        m_tot = np.append(m_tot,m_storage)
               
               
                del binned_df
            
            #get mergers ----------------------------------------------------------------------------------------------------------------
            
                
            # check whether current the snapshot has a the redshift just before the merger occurs.
            if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):
                
                #if particles have already been loaded in, loading them in again is not required
                decision2 = False if decision==True else True
                
                # The chosen particles from the accreting halo
                chosen_merger_particles = np.array([])
                
                decl=False
                
                # the index where the merger's redshift matches the redshift of the snapshot
                # we perform the selection one redhshift after - so that the accretion has definately taken place
                t_id = int(np.where(z_set_vals==red_all[i+1])[0][0])
                
                print('chosen merger particles ----------------------------------------------',len(chosen_merger_particles))
                
                #loop over the merging halos and collect particles from each of them
                for hDM in hmerge_added[t_id][0]:
                    
                    gc.collect()
                    print('halo:',hDM)

                    if (np.isin('dm_mass_profile',hDM.keys())):
                        
                        prob_occupied = calculate_poccupied(hDM,occupation_fraction)
                        print('successfully calculated poccupied')
                        
                    else:
                        print("poccupied couldn't be calculated --> dm mass profile unavailable in tangos db")
                        continue
                    
                    if (np.random.random() > prob_occupied):
                        print('skipped in accordance with occupation fraction selected')
                        continue
                                                                                                                                                                            
                    try:
                        # loading in the properties of the halo from darklight as above
                        t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,mergers=True,poccupied=occupation_fraction)
                        print(len(mstar_merging))
                        
                    except Exception as e :
                        print(e)
                        print('there are no darklight stars')
                        continue
                    
                    # if no stellar mass is expected in the halo, then selection is not performed
                    if len(mstar_merging)==0:
                        continue

                    mass_select_merge= mstar_merging[-1]/1112
                    
                    if int(mass_select_merge)<1:
                        leftover+=mstar_merging[-1]
                        continue
                    
                    # if the particle data is to be loaded in abd halo is predicted to have stellar mass
                    if float(mass_select_merge) >=1 and decision2==True:
                        
                        # try to load in the data from this snapshot
                        try:
                            DMOparticles = pynbody.load(simfn).d
                            DMOparticles.physical_units()
                            print('loaded data in mergers')
                        
                        # where this data isn't available, notify the user.
                        except:
                            print('--> DMO particle data exists but failed to read it, skipping!')
                            continue
                            
                        #reset the decision booleans (data has been loaded in now)
                        decision2 = False
                        decl=True
                        
                    
                    if int(mass_select_merge) > 0:

                        # load in the pynbody halo object of the main halo and center snapshot on it
                        try:
                            h_merge = DMOparticles.halos()[int(hDM.calculate('halo_number()'))-1]
                            pynbody.analysis.halo.center(h_merge,mode='hyb')
                                                                                            
                            r200_merge = hDM['r200c']
                        except:
                            continue

                        a_accreted_check = plum_const(hDM,red_all[i],'accreted',r200_merge)
                        
                        accreted_particles_within_selection_distance = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= 10*a_accreted_check ]
                        
                        if len(accreted_particles_within_selection_distance)==0:
                            print('no particles in the selection radius')
                            continue
                                                                         
                                        
                        # Bin the particles of merging halo
                        binned_df_merger,bins_merge,a_merge,ignored_array,sm_mer = prod_binned_df(red_all[i], mstar_merging, mstar_merging[-1], chosen_parts, DMOparticles, hDM,'accreted',np.array([]),r200_merge)
                        
                        # choose particles from each bin as decided by the plummer profile
                        choose_parts_merger,output_num_merge,tgyr_of_choice_merge,r_of_choice_merge,p_typ_merge,a_storage_merge,m_storage_merge = get_bins(bins_merge, binned_df_merger, mstar_merging[-1], a_merge,np.array([]), 0, red_all, t_all, i, 'accreted',sm_mer)

                        # if particles are choosen
                        if len(choose_parts_merger)>0:
                            # store the following values
                            
                            output_number = np.append(output_number,output_num_merge)
                                                    
                            chosen_parts = np.append(chosen_parts,choose_parts_merger)

                            redshift_of_choice = np.append(redshift_of_choice, r_of_choice_merge)
                                
                            time_of_choice = np.append(time_of_choice,tgyr_of_choice_merge)
                                
                            part_typ = np.append(part_typ,p_typ_merge)
                                
                            a_coeff_tot = np.append(a_coeff_tot, a_storage_merge)
                                
                            m_tot = np.append(m_tot, m_storage_merge)

                        
                            print('triggered , -------------------------------------------------------------------------------------- we selected',len(choose_parts_merger),'particles ------------')
                 
            # If particle data has been loaded in, delete this before the next snapshot is analysed
            if decision==True or decl==True:
                del DMOparticles
            
            
            print("Done with iteration",i)
        print("total stellar mass selected ------------------------------------------------------------------------------------------------------------------------",mstar_selected_total)
        
        print('-------------------------------------------------------------------Writing output to csv ----------------------------------------------------------------')

        
        
        df_spatially_tagged_particles = pd.DataFrame({'iords':chosen_parts , 'z':redshift_of_choice, 't':time_of_choice, 'type':part_typ, 'mstar':1112*np.ones(len(chosen_parts))})
        
        df_spatially_tagged_particles.to_csv(filename_for_run)
        
    return df_spatially_tagged_particles
    
    
    
    
def rhalf2D_dm(particles,n,r):
    #Calculate radius that encloses half the given particles.

    #Assumes each particle positions have been centered on main halo.  Adopts
    #same 'luminosity' for each particle.  Creates list of projected distances
    #from halo center (as seen from the z-axis), sorts this to get the distances
    #in increasing order, then choses the distance that encloses half the particles.
    
    rproj = sqrt(particles['x']**2 + particles['y']**2)
    
    rproj2 = [i for i in rproj if i<(r/n)]
    rproj2.sort()

    if round(len(rproj2)/2)>0:
        return rproj2[ round(len(rproj2)/2) ]
    else:
        return np.nan
                            

def calc_3D_cm(particles,masses):

    x_cm = sum(particles['x']*masses)/sum(masses)
        
    y_cm = sum(particles['y']*masses)/sum(masses)

    z_cm = sum(particles['z']*masses)/sum(masses)
                
    return np.asarray([x_cm,y_cm,z_cm])


def spatial_calculate_reffs(sim_name, particles_tagged,reffs_fname,from_file = False,from_dataframe=False,save_to_file=True):

    if (from_file == from_dataframe):
        print('please specify whether particle data is to be read from file or dataframe')
        print('from_file == True requires that string filename is provided')
        print('from_dataframe == True reuires a pandas dataframe to be named')
        print('help() prints usage instructions')
    
    tangos_path_edge     = config.get_path("tangos_path")
    tangos_path_chimera  = config.get_path("tangos_path")
    pynbody_path_edge    = config.get_path("pynbody_path")
    pynbody_path_chimera = config.get_path("pynbody_path")
    pynbody_edge_gm = config.get_path("pynbody_path")

    '''
    List of available halos
    
    'Halo383_fiducial','Halo383_fiducial_late', 'Halo383_fiducial_288', 'Halo383_fiducial_early','Halo383_Massive',
    'Halo600_fiducial', 'Halo600_fiducial_later_mergers','Halo605_fiducial','Halo624_fiducial','Halo624_fiducial_higher_finalmass',
    'Halo1445_fiducial,'Halo1459_fiducial_Mreionx02', 'Halo1459_fiducial_Mreionx03', 'Halo1459_fiducial_Mreionx12',
    'Halo600_RT', 'Halo605_RT', 'Halo624_RT', 'Halo1445_RT', 'Halo1459_RT'
    
    '''

    sims = [str(sim_name)]
    for isim,simname in enumerate(sims):
        
        print('==================================================')
        print(simname)
        
        # assign it a short name
        split = simname.split('_')
        shortname = split[0][4:]
        halonum = shortname[:]
        if len(split) > 2:
            if   halonum=='332': shortname += 'low'
            elif halonum=='383': shortname += 'late'
            elif halonum=='600': shortname += 'lm'
            elif halonum=='624': shortname += 'hm'
            elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
            elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
            elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
            else:
                print('unsupported simulation',simname,'! Not sure what shortname to give it. Aborting...')
                continue
        elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'
        #DMOname = 'Halo'+halonum+'_DMO' if split[-1]=='fiducial' else None

        #DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))

        if simname[-3] == 'x':
            DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

        else:
            DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))
        
        # set the correct paths to data files
        if halonum == '383':
            tangos_path  = tangos_path_chimera
            pynbody_path = pynbody_path_chimera #if halonum == shortname else pynbody_edge_gm
        else:
            tangos_path  = tangos_path_edge
            pynbody_path = pynbody_path_edge if halonum == shortname else pynbody_edge_gm
        
        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO particle does not data exists, skipping!')
            continue
        
     
        ## the for loop should run from here
        DMOsim,main_halo,halonums,outputs = load_indexing_data_and_halo_object(DMOname,1)
        


        red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
        #np.array([ DMOsim.timesteps[i].__dict__['redshift'] for i in range(len(DMOsim.timesteps)) ])
        t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]

        #load in the two files containing the particle data
        data_particles = pd.read_csv(particles_tagged) if from_file==True else particles_tagged
        data_redshift = data_particles['z']
        
        stored_reff = np.array([])
        stored_reff_acc = np.array([])
        stored_reff_z = np.array([])
        stored_time = np.array([])
        kravtsov_r = np.array([])
        stored_reff_tot = np.array([])
        
        for i in range(len(outputs)):

            gc.collect()
            
            selected_iords_tot = np.array(data_particles['iords'][data_particles['z']>=red_all[i]])

            selected_iords_insitu = np.array(data_particles['iords'][data_particles['z']>=red_all[i]][data_particles['type']=='insitu'])

            selected_iords_acc = np.array(data_particles['iords'][data_particles['z']>=red_all[i]][ data_particles['type']=='accreted'])
            
            #get the main halo object at the given timestep if its not available then inform the user.
            hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))
            print(hDMO)
            
            #for  the given path,entry,snapshot at given index generate a string that includes them
            simfn = join(pynbody_path,DMOname,outputs[i])
        
            # try to load in the data from this snapshot
            try:  
                DMOparticles = pynbody.load(simfn)
           
            # where this data isn't available, notify the user.
            except:
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
            
            # once the data from the snapshot has been loaded, .physical_units()
            # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
            DMOparticles.physical_units()
            
            try:
                DMOparticles['pos'] -= hDMO['shrink_center']
            except:
                print('Tangos shrink center unavailable!')
                continue
                        
            particle_selection_tot = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else []
            particle_selection_reff_insitu = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_insitu)] if len(selected_iords_insitu)>0 else []
            particle_selection_reff_acc = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_acc)] if len(selected_iords_acc)>0 else []
            
            if len(selected_iords_tot)>0:
                particle_selection_tot['pos'] -= calc_3D_cm(particle_selection_tot,particle_selection_tot['mass'])
            
            if (len(particle_selection_reff_insitu)+len(particle_selection_reff_acc))==0:
                print('skipped!')
                continue
            
            if len(particle_selection_reff_insitu) == 0:
                if len(particle_selection_reff_acc) >0:
                    
                    effective_radius_insitu = np.nan
                    effective_radius_acc = rhalf2D_dm(particle_selection_reff_acc,1,hDMO['r200c'])
            
                    stored_reff_acc = np.append(stored_reff_acc,effective_radius_acc)
                    stored_reff = np.append(stored_reff,np.nan)
                    stored_reff_z = np.append(stored_reff_z,red_all[i])
                    stored_time = np.append(stored_time, time_all[i])
                    stored_reff_tot = np.append(stored_reff_tot,effective_radius_acc)
                    kravtsov = hDMO['r200c']*0.02
                    kravtsov_r = np.append(kravtsov_r,kravtsov)

            if len(particle_selection_reff_acc) == 0:
                if len(particle_selection_reff_insitu) >0:
                    effective_radius_acc = np.nan
                    effective_radius_insitu = rhalf2D_dm(particle_selection_reff_insitu,5,hDMO['r200c'])
                    stored_reff = np.append(stored_reff,effective_radius_insitu)
                    stored_reff_acc = np.append(stored_reff_acc,np.nan)
                    stored_reff_z = np.append(stored_reff_z,red_all[i])
                    stored_time = np.append(stored_time, time_all[i])
                    stored_reff_tot = np.append(stored_reff_tot, effective_radius_insitu)
                    kravtsov = hDMO['r200c']*0.02
                    kravtsov_r = np.append(kravtsov_r,kravtsov)

            else:
                
                effective_radius_insitu = rhalf2D_dm(particle_selection_reff_insitu,5,hDMO['r200c'])
                effective_radius_acc = rhalf2D_dm(particle_selection_reff_acc,1,hDMO['r200c'])
             
                effective_tot = rhalf2D_dm(particle_selection_tot,5,hDMO['r200c'])
                
                stored_reff = np.append(stored_reff,effective_radius_insitu)

                stored_reff_acc = np.append(stored_reff_acc,effective_radius_acc)
                
                stored_reff_tot = np.append(stored_reff_tot,effective_tot)
                stored_reff_z = np.append(stored_reff_z,red_all[i])
                stored_time = np.append(stored_time, time_all[i])

                kravtsov = hDMO['r200c']*0.02
                kravtsov_r = np.append(kravtsov_r,kravtsov)
            
            print(effective_radius_insitu,effective_radius_acc)

            del DMOparticles
            del hDMO
            del particle_selection_reff_insitu
            del particle_selection_reff_acc
            
            
        #open('reffs_new23_'+halonum+'.csv','w').close()

        print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

        df_reff = pd.DataFrame({'reff':stored_reff_tot,'reff_insitu':stored_reff,'reff_acc':stored_reff_acc , 'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
                
        df_reff.to_csv(reffs_fname) if save_to_file==True else print('reffs not saved to file, to store values set save_to_file = True')
        
    return df_reff
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

