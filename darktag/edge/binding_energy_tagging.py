
# parent = pynbody_analysis.py created 2021.08.11 by Stacy Kim

# selection script = created 2021.08.21 by Sushanta Nigudkar 

'''
An EDGE specific version of theangular momentum based particle tagging script. 
Depdencies include the general tagging function as well as analysis functions defined in darktrace/tagging

'''
#import tracemalloc
#from memory_profiler import profile
#import csv
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
import darktag.tagging.binding_energy_tagging as dtag
from darktag.tagging.utils import *
from darktag.analysis.calculate import *
from ...config import config


def get_child_iords(halo,halo_catalog,DMOstate='fiducial'):

    '''
    
    Given a halo object from an AHF (Amiga's Halo Finder)
    halo catalogue, the function returns a list of dark matter and star particle id's  
    of particles belonging to 'child' or sub-halo of the main halo. 
    
    '''
    children_dm = np.array([])

    children_st = np.array([])

    sub_halonums = np.array([])

    if (np.isin('children',list(halo.properties.keys())) == True) :

        children_halonums = halo.properties['children']

        sub_halonums = np.append(sub_halonums,children_halonums)

        #print(children_halonums)                                                                                                                                                                                                                              


        for child in children_halonums:

            if (len(halo_catalog[child].dm['iord']) > 0):

                children_dm = np.append(children_dm,halo_catalog[child].dm['iord'])



            if DMO_state == 'fiducial':

                if (len(halo_catalog[child].st['iord']) > 0 ):

                    children_st = np.append(children_st,halo_catalog[child].st['iord'])

            if (np.isin('children',list(halo_catalog[child].properties.keys())) == True) :

                dm_2nd_gen,st_2nd_gen,sub_halonums_2nd_gen = get_child_iords(halo_catalog[child],halo_catalog,DMOstate)

                children_dm = np.append(children_dm,dm_2nd_gen)
                children_st = np.append(children_st,st_2nd_gen)
                sub_halonums = np.append(sub_halonums,sub_halonums_2nd_gen)
            #else:                                                                                                                                                                                                                                             
            #    print("there were no star or dark-matter iord arrays")                                                                                                                                                                                        

    #else:                                                                                                                                                                                                                                                     
    #    print("did not find children in halo properties list")                                                                                                                                                                                                

    return children_dm,children_st,sub_halonums



pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

def BE_tag_particles_edge(sim_name,occupation_fraction,fmb_percentage,PE_file = None,AHF_centers_file=None,mergers = True,AHF_centers_supplied=False,machine='astro',physics='edge1'):
    
    '''
    Function that tags particles based on angular momentum for DMOs in the EDGE suite.  

    sim_name = String that specifies the simulation name eg. Halo1459_DMO
    occupation_fraction = String, one of edge1,nadler20,edge1_rt - for darklight (specifies prob of having stars at each halo mass)
    fmb_percentage = value of f_tag or free parameter 
    mergers = boolean specifying whether to tag accreting halos 
    machine = string, one of Astro or Dirac
    
    
    Halos Available on Astro server Surrey

    'Halo383_fiducial'
    'Halo383_fiducial_late',   'Halo383_fiducial_288', 'Halo383_fiducial_early' 'Halo383_Massive'
    'Halo600_fiducial','Halo600_fiducial_later_mergers','Halo1445_fiducial'
    'Halo605_fiducial','Halo624_fiducial','Halo624_fiducial_higher_finalmass','Halo1459_fiducial',
    'Halo605_fiducial','Halo1459_fiducial_Mreionx02'#, 'Halo1459_fiducial_Mreionx03', 'Halo1459_fiducial_Mreionx12','Halo600_RT', 'Halo605_RT', 'Halo624_RT',
    'Halo1445_RT','Halo1459_RT'

    '''
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
    sims = [str(sim_name)]
    '''
    # open the file in the write mode
    with open(, 'w') as particle_storage_file:
        # create the csv writer
        writer = csv.writer(particle_storage_file)
        header = ['iords','mstar','t','z','type']
        # write a row to the csv file
        writer.writerow(header)
    '''
    
    

        
    # iterating over all the simulations in the 'sims' list
    for isim,simname in enumerate(sims):

        print('==================================================')
        print(simname)

        # assign it a short name
        split = simname.split('_')
        DMOstate = split[1]
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

        DMOname = simname 

        
        if machine == 'dirac':
            tangos_path = config.get_path("tangos_path")
            
            if physics == 'edge1':
                pynbody_path = config.get_path("pynbody_path")
    
            elif physics=='edge2':
                pynbody_path = config.get_path("pynbody_path")

        
# set the correct paths to data files
        
        if machine == 'astro':
            if halonum == '383':
                tangos_path  = config.get_path("tangos_path")
                pynbody_path = config.get_path("pynbody_path") 
            else:
                tangos_path  = config.get_path("tangos_path")
                pynbody_path = config.get_path("pynbody_path")
            

        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO simulation with name '+DMOname+' does not exist, skipping!')
            continue
        
        # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
        # load_tangos_data is a part of the 'utils.py' file in the tagging dir, it loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
        # here haloidx_at_end or 0 here specifies the index associated with the main halo at the last snapshot in the tangos db's halo catalogue
        
        DMOsim,main_halo,halonums,outputs = load_indexing_data(DMOname,1,machine=machine,physics=physics)
        
        print('HALONUMS:---',len(halonums), "OUTPUTS---",len(outputs))
        
        # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
        #t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers = False,occupation='edge1', pre_method='fiducial_with_turnover', post_scatter_method='flat')

        #calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
        zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
        
        # The redshifts and times (Gyr) of all snapshots of the given simulation from the tangos database
        red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
        #np.array([ DMOsim.timesteps[i].__dict__['redshift'] for i in range(len(DMOsim.timesteps)) ])
        t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
        #np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

        if ( len(red_all) != len(outputs) ) : 

            print('output array length does not match redshift and time arrays')


        df_tagged_particles,l_sel = dtag.BE_tag_over_full_sim_recursive(DMOsim,-1, 1, free_param_value = fmb_percentage,PE_file=PE_file,pynbody_path  = pynbody_path, occupation_frac = 'edge1')
        
        #df_tagged_particles = dtrace.BE_tag_over_full_sim(DMOsim, free_param_value = 0.01, PE_file=PE_file,pynbody_path  = pynbody_path, occupation_frac = 'all')
        return df_tagged_particles


def calc_3D_cm(particles,masses):
    
    x_cm = sum(particles['x']*masses)/sum(masses)
        
    y_cm = sum(particles['y']*masses)/sum(masses)
    
    z_cm = sum(particles['z']*masses)/sum(masses)

    return np.asarray([x_cm,y_cm,z_cm])


def center_on_tagged(radial_dists,mass):
    masses = np.asarray(mass)
        
    return sum(radial_dists*masses)/sum(masses)



def angmom_calculate_reffs(sim_name, particles_tagged,reffs_fname,AHF_centers_file=None,from_file = False,from_dataframe=False,save_to_file=True,AHF_centers_supplied=False,machine='astro',physics='edge1'):

    '''

    'Halo383_fiducial'
    'Halo383_fiducial_late', 'Halo383_fiducial_288', 'Halo383_fiducial_early','Halo383_Massive',
    'Halo600_fiducial','Halo600_fiducial_later_mergers','Halo605_fiducial','Halo624_fiducial',
    'Halo624_fiducial_higher_finalmass','Halo1445_fiducial','Halo1445_fiducial','Halo1459_fiducial_Mreionx02', 'Halo1459_fiducial_Mreionx03','Halo1459_fiducial_Mreionx12','Halo600_RT', 'Halo605_RT', 'Halo624_RT',
    'Halo1445_RT','Halo1459_RT'

    '''
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                    

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
       # DMOname = 'Halo'+halonum+'_DMO' if split[-1]=='fiducial' else None
        #DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))

        if simname[-3] == 'x':
            DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

        else:
            DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:]))) #if split[1]=='fiducial' else None

                        
# set the correct paths to data files
        if machine=='astro':
            tangos_path = config.get_path("tangos_path")
            pynbody_path = config.get_path("pynbody_path")

        if machine == 'dirac':
            tangos_path = config.get_path("tangos_path")
            pynbody_path = config.get_path("pynbody_path")
            
        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO particle does not data exists, skipping!')
            continue
        
        DMOsim,main_halo,halonums,outputs = load_indexing_data(DMOname,1,machine=machine,physics=physics)
        
        #outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])[-len(halonums):]

        print(outputs)
        
        snapshots = [ f for f in listdir(pynbody_path+DMOname) if (isdir(join(pynbody_path,DMOname,f)) and f[:6]=='output') ]
        
        #sort the list of snapshots in ascending order

        snapshots.sort()

        
        red_all =  main_halo.calculate_for_progenitors('z()')[0][::-1]
        #np.array([DMOsim.timesteps[i].__dict__['redshift'] for i in range(len(DMOsim.timesteps)) ])
        t_all =  main_halo.calculate_for_progenitors('t()')[0][::-1]
        #np.array([DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

        #load in the two files containing the particle data
        if ( len(red_all) != len(outputs) ) : 
            print('output array length does not match redshift and time arrays')

        data_particles = pd.read_csv(particles_tagged)

        #print('data parts',data_particles['t'])

        data_t = np.asarray(data_particles['t'].values)
        
        stored_reff = np.array([])
        stored_reff_acc = np.array([])
        stored_reff_z = np.array([])
        stored_time = np.array([])
        kravtsov_r = np.array([])
        stored_reff_tot = np.array([])
        KE_energy = np.array([])
        PE_energy = np.array([])
        lum_based_halflight = np.array([])

        AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_supplied == True else None
                
        for i in range(len(outputs)):

            gc.collect()

            
            if len(np.where(data_t <= float(t_all[i]))) == 0:
                continue

            
            dt_all = data_particles[data_particles['t']<=t_all[i]]

            
            data_grouped = dt_all.groupby(['iords']).sum()
            

            selected_iords_tot = data_grouped.index.values

            data_insitu = dt_all[dt_all['type'] == 'insitu'].groupby(['iords']).sum()
            
            selected_iords_insitu_only = data_insitu.index.values
            
            
            if selected_iords_tot.shape[0]==0:
                continue

            
            mstars_at_current_time = data_grouped['mstar'].values
            
            half_mass = float(mstars_at_current_time.sum())/2
            
            print(half_mass)
            #get the main halo object at the given timestep if its not available then inform the user.

           
            hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))
                
            print(hDMO)
                
            #for  the given path,entry,snapshot at given index generate a string that includes them
            simfn = join(pynbody_path,DMOname,outputs[i])
            
            # try to load in the data from this snapshot
            try:  DMOparticles = pynbody.load(simfn)

            # where this data isn't available, notify the user.
            except:
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
            
            # once the data from the snapshot has been loaded, .physical_units()
            # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
            DMOparticles.physical_units()

            

            try:
                if AHF_centers_supplied==False:
                    h = DMOparticles.halos()[int(halonums[i])-1]
                    
                elif AHF_centers_supplied == True:
                    pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                    
                    
                    AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                        
                    h = DMOparticles.halos()[int(AHF_crossref)] 
                            
                    children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                            
                    child_str_l = children_ahf[0][1:-1].split()

                    children_ahf_int = list(map(float, child_str_l))
                    
                    
                    halo_catalogue = DMOparticles.halos()
                    
                    subhalo_iords = np.array([])
                        
                    for i in children_ahf_int:
                                
                        subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(i)].dm['iord'])
                                                                                                                                                 
                    h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
                    

                    
                pynbody.analysis.halo.center(h)

            except:
                print('centering data unavailable')
                continue


            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')

            except:
                print('could not calculate R200c')
                continue
            
            DMOparticles = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ]        

            particle_selection_reff_tot = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else []

            particles_only_insitu = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_insitu_only)] if len(selected_iords_insitu_only) > 0 else []

            print('m200 value---->',hDMO['M200c'])
            
            if (len(particle_selection_reff_tot))==0:
                print('skipped!')
                continue
            else:

        
                masses = [ data_grouped.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

                masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                    
                cen_stars = calc_3D_cm(particles_only_insitu,masses_insitu)
                
                particle_selection_reff_tot['pos'] -= cen_stars
                
                # new cutoff calc begins 
                distances = np.sqrt(particle_selection_reff_tot['x']**2+particle_selection_reff_tot['y']**2) #+ particle_selection_reff_tot['z']**2)                
                            
                idxs_distances_sorted = np.argsort(distances)

                sorted_distances = np.sort(distances)

                distance_ordered_iords = np.asarray(particle_selection_reff_tot['iord'][idxs_distances_sorted])
                
                print('array lengths',len(set(distance_ordered_iords)),len(distance_ordered_iords))

                sorted_massess = [data_grouped.loc[n]['mstar'] for n in distance_ordered_iords]

                cumilative_sum = np.cumsum(sorted_massess)

                R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]

                lum_for_each_part = produce_lums_grouped( dt_all, particle_selection_reff_tot['iord'], t_all[i])
                hlight_r = calc_halflight(particle_selection_reff_tot, lum_for_each_part, band='v', cylindrical=False)
                
                print(hlight_r)
                
                lum_based_halflight = np.append(lum_based_halflight,hlight_r)
                
                stored_reff_z = np.append(stored_reff_z,red_all[i])
                stored_time = np.append(stored_time, t_all[i])
                   
                stored_reff = np.append(stored_reff,float(R_half))
                kravtsov = hDMO['r200c']*0.02
                kravtsov_r = np.append(kravtsov_r,kravtsov)

                particle_selection_reff_tot['pos'] += cen_stars

                print('halfmass radius:',R_half)
                print('Kravtsov_radius:',kravtsov)
                
            

        print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

        df_reff = pd.DataFrame({'halflight':lum_based_halflight, 'reff':stored_reff, 'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
        
        #df2_reff = pd.DataFrame({'z_tangos':ztngs, 't_tangos':ttngs,'reff_tangos':hlftngs})
        
        df_reff.to_csv(reffs_fname) if save_to_file==True else print('reffs not saved to file, to store values set save_to_file = True')
        #df2_reff.to_csv('reffs_new22_tangos'+halonum+'.csv')
        print('wrote', reffs_fname)
        
    return df_reff
