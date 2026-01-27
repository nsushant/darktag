
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
import darktag.tagging.angular_momentum_tagging as dtag
from darktag.tagging.utils import *
from darktag.analysis.calculate import *
from sklearn.cluster import DBSCAN
from collections import Counter
from ...config import config

def get_child_iords(halo,dmo_particles,DMO_state='fiducial'):

    pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]

    '''
    
    Given a halo object from an AHF (Amiga's Halo Finder)
    halo catalogue, the function returns a list of dark matter and star particle id's  
    of particles belonging to 'child' or sub-halo of the main halo. 
    
    '''
    halo_catalog = dmo_particles.halos(halo_numbers="v1")

    print(halo_catalog)
    print(halo_catalog.keys())

    children_dm = np.array([])

    children_st = np.array([])
    
    children_gas = np.array([])

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
                    '''
                    try: 
                        children_gas = np.append(children_gas,halo_catalog[child].g['iord'])
                    except:
                        print("No gas")
                        pass
                    '''
            #if (np.isin('children',list(halo_catalog[child].properties.keys())) == True) :

              #  dm_2nd_gen,st_2nd_gen,sub_halonums_2nd_gen = get_child_iords(halo_catalog[child],halo_catalog,DMO_state)

              #  children_dm = np.append(children_dm,dm_2nd_gen)
              #  children_st = np.append(children_st,st_2nd_gen)
              #  sub_halonums = np.append(sub_halonums,sub_halonums_2nd_gen)
            #else:                                                                                                                                                                                                                                             
            #    print("there were no star or dark-matter iord arrays")                                                                                                       '''                                               
                                              
        
    #else:                                                                                                                                                                                                                                                     
    #    print("did not find children in halo properties list")                                                                                                                                                               
                                 

    return children_dm,children_st,sub_halonums








pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

def angmom_tag_particles_edge(sim_name,halo_number=1,mergers = True, machine='dirac',physics='edge1',recursive=True):
    
    '''
    Function that tags particles based on angular momentum for DMOs in the EDGE suite.  

    sim_name = String that specifies the simulation name eg. Halo1459_DMO
    occupation_fraction = String, one of edge1,nadler20,edge1_rt - for darklight (specifies prob of having stars at each halo mass)
    fmb_percentage = value of f_tag or free parameter 
    mergers = boolean specifying whether to tag accreting halos 
    machine = string, one of Astro or Dirac
    
    '''  

    print(sim_name)

    # assign it a short name
    split = sim_name.split('_')
    DMOstate = split[1]
    shortname = split[0][4:]
    halonum = shortname[:]
        
    DMOname = sim_name 

    '''
    # get particle data at z=0 for DMO sims, if available
    if DMOname==None:
        print('--> DMO simulation with name '+DMOname+' does not exist, skipping!')
        continue
    ''' 
    # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
    # load_tangos_data is a part of the 'utils.py' file in the tagging dir, it loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
    # here haloidx_at_end or 0 here specifies the index associated with the main halo at the last snapshot in the tangos db's halo catalogue
    tangos.init_db(join(config.get_path("tangos_path"),str(split[0]+".db")))
    
    DMOsim = tangos.get_simulation(DMOname)

    t_all,red_all,main_halo,halonums,outputs = load_indexing_data(DMOsim,halo_number)
        
    print('HALONUMS:---',len(halonums), "OUTPUTS---",len(outputs))

    if (config.get_path("manual_halonum_path")==""):

        hnumpath = None 

    else:
        
        hnumpath = config.get_path("manual_halonum_path")

    if recursive==True:
        df_tagged_particles,l_sel = dtag.angmom_tag_over_full_sim_recursive(DMOsim,-1, halo_number, free_param_value = config.get("tagging","ftag"), pynbody_path  = os.path.join(config.get_path('pynbody_path'),str(sim_name)),AHF_centers_filepath=hnumpath)
        
    else:
        df_tagged_particles = dtag.angmom_tag_over_full_sim(DMOsim, halonumber=halo_number, free_param_value = config.get("tagging","ftag"), particle_storage_filename=None, mergers=mergers)
    
    return df_tagged_particles


def calc_3D_cm(particles,masses):
    
    x_cm = sum(particles['x']*masses)/sum(masses)
        
    y_cm = sum(particles['y']*masses)/sum(masses)
    
    z_cm = sum(particles['z']*masses)/sum(masses)

    return np.asarray([x_cm,y_cm,z_cm])


def center_on_tagged(radial_dists,mass):
    masses = np.asarray(mass)
        
    return sum(radial_dists*masses)/sum(masses)



def angmom_calculate_reffs(sim_name, particles_tagged,reffs_fname,from_file = False,from_dataframe=False,save_to_file=True,machine='dirac',physics='edge1'):

     
    if (config.get_path("manual_halonum_path") == ""):

        path_AHF_halonums= None 
    else:
        path_AHF_halonums = join(config.get_path("manual_halonum_path"),"DMO",sim_name)  
    
    AHF_halonums = None

    if os.path.isfile(path_AHF_halonums): 

        AHF_halonums = pd.read_csv(path_AHF_halonums) 

        if len(AHF_halonums['snapshot']) > 0:
            print("Using AHF catalogue======================================================")
            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]            

        else:
            print("AHF halonums file at "+path_AHF_halonums+" is empty, using HOP catalogue")
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    else: 
        print("AHF halonumsfile at"+path_AHF_halonums+" does not exist, using HOP catalogue")
        pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
    
  
    # assign it a short name
    split = sim_name.split('_')
    shortname = split[0][4:]
    halonum = shortname[:]

    if simname[-3] == 'x':
        DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

    else:
        DMOname = simname
    
    
    tangos.init_db(join(config.get_path("tangos_path"),str(split[0]+".db")))
    
    DMOsim = tangos.get_simulation(DMOname)
        
    t_all,red_all,main_halo,halonums,outputs = load_indexing_data(DMOsim,1,machine=machine,physics=physics)
    
    red_all =  main_halo.calculate_for_progenitors('z()')[0][::-1]
    t_all =  main_halo.calculate_for_progenitors('t()')[0][::-1]
        
        
    if ( len(red_all) != len(outputs) ):
        print('output array length does not match redshift and time arrays')
        data_particles = pd.read_csv(particles_tagged)
        data_t = np.asarray(data_particles['t'].values)
        return 0 

        
    stored_reff = np.array([])
    stored_reff_acc = np.array([])
    stored_reff_z = np.array([])
    stored_time = np.array([])
    kravtsov_r = np.array([])
    stored_reff_tot = np.array([])
    KE_energy = np.array([])
    PE_energy = np.array([])
    lum_based_halflight = np.array([])

    PrevBGMMIords = np.array([]) 
        
        
    for i in range(len(outputs)):

        gc.collect()

        if len(np.where(data_t <= float(t_all[i]))) == 0:
            continue

            
        dt_all = data_particles[data_particles['t']<=t_all[i]]

            
        data_grouped = dt_all.groupby(['iords']).sum()
            

        selected_iords_tot = data_grouped.index.values

            
        if selected_iords_tot.shape[0]==0:
            continue
            
        mstars_at_current_time = data_grouped['mstar'].values
            
        half_mass = float(mstars_at_current_time.sum())/2
            
        print(half_mass)
            #get the main halo object at the given timestep if its not available then inform the user.

           
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))
                
        print(hDMO)
            
        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
        if type(AHF_halonums) == type(None):
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        #for  the given path,entry,snapshot at given index generate a string that includes them
        simfn = join(config.get_path("pynbody_path"),DMOname,outputs[i])
            
        # try to load in the data from this snapshot
        try:  
            DMOparticles = pynbody.load(simfn)

        # where this data isn't available, notify the user.
        except Exception as err_load:
            print('--> DMO particle data exists but failed to read it, skipping!',err_load)
            continue
            
        # once the data from the snapshot has been loaded, .physical_units()
        # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
        #DMOparticles.physical_units()
            
        children_dm = np.array([])

        children_st = np.array([])

        try:
            #if AHF_centers_supplied==False:
                    
            if type(AHF_halonums) != type(None):
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                print('halonums cat', DMOparticles.halos(halo_numbers='v1'),DMOparticles.halos(halo_numbers='v1').keys())
                #pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]

                halonum_snap = AHF_halonums[AHF_halonums["snapshot"] == str(outputs[i])]["AHF halonum"].values
                        
                print("Halonum in snap --> ",halonum_snap)

                h = DMOparticles.halos(halo_numbers='v1')[int(halonum_snap)]
                halo_catalog = DMOparticles.halos(halo_numbers="v1")
                print(halo_catalog.keys())
                print("main halo done")
                DMO_state = "DMO"
                children_dm = np.array([])

                children_st = np.array([])

                sub_halonums = np.array([])

                if (np.isin('children',list(h.properties.keys())) == True) :

                    children_halonums = h.properties['children']

                    sub_halonums = np.append(sub_halonums,children_halonums)

                        #print(children_halonums)                                                                                                                                                                               

                    for child in children_halonums:

                        if (len(halo_catalog[child].dm['iord']) > 0):

                            children_dm = np.append(children_dm,halo_catalog[child].dm['iord'])



                        if DMO_state == 'fiducial':

                            if (len(halo_catalog[child].st['iord']) > 0 ):

                                children_st = np.append(children_st,halo_catalog[child].st['iord'])

                
            else:
                print("confirmed switch to HOP")
                pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                h = DMOparticles.halos()[int(halonums[i])-1]
                halo_catalog = DMOparticles.halos()
                

                #pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                #children_dm,children_st,sub_halonums = get_child_iords(h.dm,DMOparticles,DMO_state='DMO')
                
            DMOparticles.physical_units()    
            pynbody.analysis.halo.center(h.dm)
            #pynbody.analysis.angmom.faceon(h.dm[h.dm['r']<5])
                
        except Exception as e:
            print('centering data unavailable',e)
            continue


        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')

        except:
            print('could not calculate R200c')
            continue
            
        

        DMOparticles = DMOparticles.dm[sqrt(DMOparticles.dm['pos'][:,0]**2 + DMOparticles.dm['pos'][:,1]**2 + DMOparticles.dm['pos'][:,2]**2) <= r200c_pyn ]        

        DMOparticles_only_insitu = DMOparticles.dm[np.logical_not(np.isin(DMOparticles.dm['iord'],children_dm))]


        particle_selection_reff_tot = DMOparticles_only_insitu[np.isin(DMOparticles_only_insitu['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else [] 

        # -------- BGMM Implementation---------#
            
        x = particle_selection_reff_tot['x']
        y = particle_selection_reff_tot['y']
        z = particle_selection_reff_tot['z']
            
        if len(x)<=2 : 

            continue 
            
        xy = np.column_stack((x, y))



        labelsALL = gmm.predict(xy)
            

        dbscan = DBSCAN(eps=0.05, min_samples=2)
        dbscan.fit(xy, sample_weight = particle_selection_reff_tot['mass'])
        labelsALL = dbscan.labels_

            
        prevp = particle_selection_reff_tot[np.isin(particle_selection_reff_tot['iord'],PrevBGMMIords.flatten())]



        if (len(np.where(prevp == True)[0]) == 0):
            print("largest cluster used, No previous particles")
                #largest = np.argmax(gmm.weights_)                                                                                                                                       
                
            labels_no_noise = [label for label in labelsALL if label != -1]
                
            print("len labels_no_noise:", len(labels_no_noise))
                
            counter = Counter(labels_no_noise)
                
            largest_cluster_label, largest_cluster_size = counter.most_common(1)[0]
                
            largest = largest_cluster_label


        else:


            labels_prev = labelsALL[prevp]
                
            print("len labels_prev", len(labels_prev))
                
            labels_no_noise = [label for label in labels_prev if label != -1]
        
            counter = Counter(labels_no_noise)
            largest_cluster_label, largest_cluster_size = counter.most_common(1)[0]

            largest = largest_cluster_label




            
        particle_selection_reff_tot = particle_selection_reff_tot[np.where(labelsALL == largest)]
        PrevBGMMIords = np.delete( PrevBGMMIords, np.arange(len(PrevBGMMIords)) )
        PrevBGMMIords = np.append(PrevBGMMIords,np.asarray(particle_selection_reff_tot['iord']))
        pynbody.analysis.halo.center(particle_selection_reff_tot)
            
            
        if (len(particle_selection_reff_tot))==0:
            print('skipped!')
            continue
        else:
                
            masses = [ data_grouped.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]
                                
                
            # new cutoff calc begins 
            distances = np.sqrt(particle_selection_reff_tot['x']**2+particle_selection_reff_tot['y']**2) 
            #+ particle_selection_reff_tot['z']**2)                
                            
            idxs_distances_sorted = np.argsort(distances)

            sorted_distances = np.sort(distances)

            distance_ordered_iords = np.asarray([particle_selection_reff_tot['iord'][number] for number in idxs_distances_sorted])
                
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
                
                
            print('halfmass radius:',R_half)
            print('Kravtsov_radius:',kravtsov)
                
            

    print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

    df_reff = pd.DataFrame({'halflight':lum_based_halflight, 'reff':stored_reff, 'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
        
        
    df_reff.to_csv(reffs_fname) if save_to_file==True else print('reffs not saved to file, to store values set save_to_file = True')
    #df2_reff.to_csv('reffs_new22_tangos'+halonum+'.csv')
    print('wrote', reffs_fname)
        
    return df_reff


