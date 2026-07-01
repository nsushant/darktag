
import os
import sys
import gc
import random

import numpy as np
from numpy import sqrt
import pandas as pd

import pynbody
import tangos
from darklight import DarkLight
from os.path import join
from tangos.examples.mergers import get_mergers_of_major_progenitor

from .utils import *
from ..config import config

def rank_order_particles_by_angmom(particles):
    
    '''
    Inputs: 

    particles - Particle data from pynbody
    
    Returns: 
    
    a list of particle IDs ordered by their corresponding angular momenta.
    
    '''
    
    print('this is how many DMOparticles were passed',len(particles))

    # makes the array 1D via sqrt(jx^2 + jy^2 +jz^2)
    angular_momenta = get_dist(particles['j'])

    #values arranged in ascending order
    sorted_indicies = np.argsort(angular_momenta.flatten())

    # particle ids sorted by angular momentum
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

    # getting the 'tagging fraction' number of low angular momentum particles 
    size_of_tagging_fraction = int(particles_sorted_by_angmom.shape[0]*tagging_fraction)
    
    particles_in_tagging_fraction = particles_sorted_by_angmom[:size_of_tagging_fraction]
    
    #dividing stellar mass evenly over all the particles in the most bound fraction 
    
    print('assigning stellar mass')

    # stellar mass to be assigned per particle
    stellar_mass_assigned = float(snapshot_stellar_mass/len(list(particles_in_tagging_fraction))) if len(list(particles_in_tagging_fraction))>0 else 0
        
    array_iords = particles_in_tagging_fraction
    
    array_masses = np.repeat(stellar_mass_assigned,len(array_iords)) 

    # array that contains (particle ids, mstar contributions) for this snap of tagging
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
    


def angmom_tag_over_full_sim(DMOsim, halonumber = 1 ,free_param_value = 0.01, particle_storage_filename=None, mergers = True, AHF_centers_file=None, occupation_frac='all'):
    
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
    
    # Name of simulation
    DMOname = DMOsim.path

    t_all,red_all,main_halo,halonums,outputs = load_indexing_data(DMOsim,halonumber)

    # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total =DarkLight(main_halo,DMO=True,n=config.get("darklight","n"),mergers=False)

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

    
    # if an AHF centering file is provided use the centers stored within it
    AHF_centers = pd.read_csv(config.get_path("manual_halonum_path")) if AHF_centers_file != None else None
    df_header = pd.DataFrame({'iords':[], 'mstar':[],'t':[],'z':[],'type':[]})
    
    df_header.to_csv(particle_storage_filename,mode="w",header=True)

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids   = np.array([])
    
    tagged_iords_to_write  = np.array([])
    tagged_types_to_write  = np.array([])
    tagged_mstars_to_write = np.array([])
    
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    
    # looping over all snapshots  
    for i in range(len(outputs)):
        gc.collect()
        
        # was particle data loaded in (insitu) 
        decision=False

        # was particle data loaded in through the accreted tagging part
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
        msn = float(np.mean(np.asarray(mstar_s_insitu)[:, idrz] if np.asarray(mstar_s_insitu).ndim == 2 else np.asarray(mstar_s_insitu)[idrz]))              

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            # else msp = previous snap's mstar value
            elif idrz_previous >= 0:
                msp = float(np.mean(np.asarray(mstar_s_insitu)[:, idrz_previous] if np.asarray(mstar_s_insitu).ndim == 2 else np.asarray(mstar_s_insitu)[idrz_previous]))
        else:
            print('There is no stellar mass at current timestep')
            continue

                                                                    
        #calculate the difference in mass between the two mstar's
        mass_select = int(msn-msp)
        print('stellar mass to be tagged in this snap -->',mass_select)
        print('tagging for t (gyr) = ',t_all[i])

        # if stellar mass is to be tagged then load in particle data 
        if mass_select>0:
            
            # try to load in the data from this snapshot
            
            try:
                simfn = join(config.get_path("pynbody_path"),DMOname,outputs[i])
                
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

            # data was loaded in
            decision=True

            print('mass to be tagged insitu:',mass_select)
            
            try:
                hDMO['r200c']
            except:
                print("Couldn't load in the R200 at timestep:" , i)
                continue
        
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

            
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            
            pynbody.analysis.halo.center(h)
        
            try:                                                                                                                                                                                              
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
                                                                                                                                                                                                              
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
                                                                                                                                                                            
                        
            DMOparticles_insitu_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]

            # will only be non empty if using AHF catalog
            DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]

            
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

            row_to_write = pd.DataFrame({'iords':array_to_write[0],
                                         'mstar':array_to_write[1],
                                         't':np.repeat(t_all[i],len(array_to_write[0])),
                                         'z':np.repeat(red_all[i],len(array_to_write[0])),
                                         'type':np.repeat('insitu',len(array_to_write[0]))})

            if particle_storage_filename != None:
                row_to_write.to_csv(particle_storage_filename,mode='a',header=False)
            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
            
            #pynbody.analysis.halo.center(h,mode='hyb').revert()

            del DMOparticles_insitu_only
            
            #get mergers ----------------------------------------------------------------------------------------------------------------
            # check whether current the snapshot has a the redshift just before the merger occurs.
        
        if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):

            # do not load in data again if loaded in through the insitu tagging part
            decision2 = False if decision==True else True

            # loaded data in this snap
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
                        
                
                try:
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging =DarkLight(hDM,DMO=True,n=config.get("darklight","n"),mergers=True)

                    if len(mstar_merging)==0:
                        print("halo has not yet formed stars")
                        continue


                
                except Exception as e :
                    print(e)
                    print('there are no darklight stars')
                
                    continue
        
        
                
                mass_select_merge= mstar_merging[-1]

                print("tagging accreted Mstar = ",mass_select_merge )

              
                if int(mass_select_merge)<1:
                    leftover+=mstar_merging[-1]
                    continue
                
                
                simfn = join(config.get_path("pynbody_path"),DMOname,outputs[i])

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
                        
                    except Exception as ex:
                        print('centering data unavailable, skipping',ex)
                        continue
                                                                                                           
                    r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                    DMOparticles_acc_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn_acc] 

                                            
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
                    
                    row_to_write = pd.DataFrame({'iords':array_to_write_accreted[0], 
                                  'mstar':array_to_write_accreted[1],
                                  't':np.repeat(t_all[i],len(array_to_write_accreted[0])),
                                  'z':np.repeat(red_all[i],len(array_to_write_accreted[0])),
                                  'type':np.repeat('accreted',len(array_to_write_accreted[0]))})

                    if particle_storage_filename != None:
                        row_to_write.to_csv(particle_storage_filename,mode='a',header=False)
        
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


def angmom_tag_multi_instance(
    DMOsim,
    n_instances,
    halonumber=1,
    free_param_value=0.01,
    output_prefix=None,
    mergers=True,
    AHF_centers_file=None,
    occupation_frac='all',
    cluster_file=None,
    track_cluster_file=None,
):
    '''
    Runs n_instances independent DarkLight realisations over the full simulation,
    loading each snapshot ONCE and writing one output CSV per instance.

    Inputs:

    DMOsim           - tangos simulation object
    n_instances      - number of independent DarkLight realisations to run
    halonumber       - halo number (default 1)
    free_param_value - tagging fraction (default 0.01)
    output_prefix    - prefix for output filenames; defaults to "{sim_name}_tagged"
    mergers          - whether to include accreting/merging halos (default True)
    AHF_centers_file - path to AHF centering CSV (optional)
    occupation_frac  - occupation fraction regime: 'all', 'nadler20', 'edge1', 'edgert'

    Returns:

    list of output CSV filenames (length n_instances)
    Each CSV has columns: iords, mstar, t, z, type
    '''

    DMOname = DMOsim.path

    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(DMOsim, halonumber)

    # Load cluster iords lookup from HDF5 if supplied
    cluster_iords_map = None
    _tc_halonum_map   = None   # {snap: int halonum} populated from track_cluster_file

    if track_cluster_file is not None:
        import h5py
        _tc_data = {}
        with h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_data[snap] = {
                        'halonum': int(f[snap]['main']['halonum'][()]),
                        'iords':   f[snap]['main']['iords'][:],
                    }
        _tc_halonum_map  = {s: d['halonum'] for s, d in _tc_data.items()}
        cluster_iords_map = {s: d['iords']   for s, d in _tc_data.items()}
        print(f'Loaded track_cluster file: {len(_tc_data)} snapshots, using AHF halonums + cluster iords')
    elif cluster_file is not None:
        import h5py
        with h5py.File(cluster_file, 'r') as f:
            cluster_iords_map = {k: f[k][:] for k in f.keys()}
        print(f'Loaded cluster iords for {len(cluster_iords_map)} snapshots from {cluster_file}')

    # N independent DarkLight mass histories for the main halo — n=1 per call for genuine stochasticity
    print(f'Running DarkLight {n_instances} time(s) for main halo...')
    dl_histories = [
        DarkLight(main_halo, DMO=True, n=1, mergers=False)
        for _ in range(n_instances)
    ]
    # each entry: (t, redshift, vsmooth, sfh_insitu, mstar_s_insitu, mstar_total)

    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
    hmerge_added, z_set_vals = group_mergers(zmerge, hmerge)

    if len(red_all) != len(outputs):
        print('output array length does not match redshift and time arrays')

    AHF_centers = pd.read_csv(config.get_path("manual_halonum_path")) if AHF_centers_file is not None else None

    # Initialise N output files
    if output_prefix is None:
        output_prefix = DMOname + '_tagged'
    os.makedirs(output_prefix, exist_ok=True)
    filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
    _header = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    for fn in filenames:
        _header.to_csv(fn, mode='w', header=True)

    def _mstar_at(mstar_arr, t_dl, t_target):
        idx = np.argmin(abs(t_dl - t_target))
        arr = np.asarray(mstar_arr)
        return float(np.mean(arr[:, idx] if arr.ndim == 2 else arr[idx]))

    # Main snapshot loop
    for i in range(len(outputs)):
        gc.collect()

        print('Current snapshot -->', outputs[i])

        hDMO = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        z_val = red_all[i]
        t_val = t_all[i]

        # Compute insitu mass_select for each instance (no I/O)
        mass_selects_insitu = []
        for k in range(n_instances):
            t_dl, _, _, _, mstar_s_insitu_k, _ = dl_histories[k]
            msn = _mstar_at(mstar_s_insitu_k, t_dl, t_val)
            if msn == 0:
                mass_selects_insitu.append(0)
                continue
            msp = _mstar_at(mstar_s_insitu_k, t_dl, t_all[i - 1]) if i > 0 else 0.0
            mass_selects_insitu.append(int(msn - msp))

        merger_snap = (
            mergers
            and (i + 1 < len(red_all))
            and (red_all[i + 1] in z_set_vals)
        )
        need_snap = any(m > 0 for m in mass_selects_insitu) or merger_snap

        if not need_snap:
            print("Done with iteration", i)
            continue

        # Load snapshot ONCE
        simfn = join(config.get_path("pynbody_path"), DMOname, outputs[i])
        try:
            print(simfn)
            print('loading DMO particles')
            DMOparticles = pynbody.load(simfn)
            DMOparticles.physical_units()
        except Exception as e:
            print(e)
            print('--> failed to load snapshot, skipping')
            print("Done with iteration", i)
            continue

        # Insitu block
        if any(m > 0 for m in mass_selects_insitu):
            try:
                hDMO['r200c']
            except Exception:
                print("Couldn't load R200 at timestep:", i)
                del DMOparticles
                print("Done with iteration", i)
                continue

            subhalo_iords = np.array([])

            if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                ahf_num = _tc_halonum_map[outputs[i]]
                h = DMOparticles.halos(halo_numbers="v1")[int(ahf_num)]
            elif AHF_centers_file is None:
                h = DMOparticles.halos()[int(halonums[i]) - 1]
            else:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]
                h = DMOparticles.halos(halo_numbers="v1")[int(AHF_crossref)]
                children_ahf_int = h.properties['children']
                halo_catalogue = DMOparticles.halos(halo_numbers="v1")
                for ch in children_ahf_int:
                    if ch != AHF_crossref:
                        subhalo_iords = np.append(subhalo_iords, halo_catalogue[int(ch)].dm['iord'])
                h = h[np.logical_not(np.isin(h['iord'], subhalo_iords))] if len(subhalo_iords) > 0 else h

            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            pynbody.analysis.halo.center(h)

            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')
            except Exception:
                print('could not calculate R200c')
                del DMOparticles
                print("Done with iteration", i)
                continue

            DMOparts_insitu = DMOparticles[
                sqrt(DMOparticles['pos'][:, 0] ** 2
                     + DMOparticles['pos'][:, 1] ** 2
                     + DMOparticles['pos'][:, 2] ** 2) <= r200c_pyn
            ]
            DMOparts_insitu = DMOparts_insitu[
                np.logical_not(np.isin(DMOparts_insitu['iord'], subhalo_iords))
            ]

            if cluster_iords_map is not None and outputs[i] in cluster_iords_map:
                DMOparts_insitu = DMOparts_insitu[
                    np.isin(DMOparts_insitu['iord'], cluster_iords_map[outputs[i]])
                ]

            # Angular-momentum ranking ONCE per snap
            parts_sorted_angmom = rank_order_particles_by_angmom(DMOparts_insitu)
            del DMOparts_insitu

            # Fan across N instances
            if parts_sorted_angmom.shape[0] > 0:
                for k in range(n_instances):
                    if mass_selects_insitu[k] > 0:
                        arr = assign_stars_to_particles(
                            mass_selects_insitu[k], parts_sorted_angmom, float(free_param_value)
                        )
                        row = pd.DataFrame({
                            'iords': arr[0],
                            'mstar': arr[1],
                            't': np.repeat(t_val, len(arr[0])),
                            'z': np.repeat(z_val, len(arr[0])),
                            'type': np.repeat('insitu', len(arr[0])),
                        })
                        row.to_csv(filenames[k], mode='a', header=False)

        # Mergers block
        if merger_snap:
            t_id = int(np.where(z_set_vals == red_all[i + 1])[0][0])

            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:', hDM)

                if occupation_frac != 'all':
                    try:
                        prob_occupied = calculate_poccupied(hDM, 2.5e7)
                    except Exception as e:
                        print(e)
                        print("poccupied couldn't be calculated")
                        continue
                else:
                    prob_occupied = 1.0

                # Halo centering and particle ranking done ONCE (deterministic given snap)
                try:
                    h_merge = DMOparticles.halos()[int(hDM.calculate('halo_number()')) - 1]
                    pynbody.analysis.halo.center(h_merge.dm)
                    r200c_pyn_acc = pynbody.analysis.halo.virial_radius(
                        h_merge.d, overden=200, r_max=None, rho_def='critical'
                    )
                except Exception as ex:
                    print('centering data unavailable, skipping', ex)
                    continue

                DMOparts_acc = DMOparticles[
                    sqrt(DMOparticles['pos'][:, 0] ** 2
                         + DMOparticles['pos'][:, 1] ** 2
                         + DMOparticles['pos'][:, 2] ** 2) <= r200c_pyn_acc
                ]

                try:
                    acc_sorted = rank_order_particles_by_angmom(DMOparts_acc)
                except Exception:
                    del DMOparts_acc
                    continue
                del DMOparts_acc

                # Each instance gets its own DarkLight draw (n=1) and occupation check
                for k in range(n_instances):
                    if occupation_frac != 'all' and np.random.random() > prob_occupied:
                        print('Skipped instance', k)
                        continue

                    try:
                        _, _, _, _, _, mstar_merging_k = DarkLight(
                            hDM, DMO=True, n=1, mergers=True
                        )
                        if np.asarray(mstar_merging_k).size == 0:
                            continue
                    except Exception as e:
                        print(e, '-- skipping instance', k)
                        continue

                    mass_merge_k = float(np.asarray(mstar_merging_k).flat[-1])
                    if int(mass_merge_k) < 1:
                        continue

                    arr = assign_stars_to_particles(
                        int(mass_merge_k), acc_sorted, float(free_param_value)
                    )
                    row = pd.DataFrame({
                        'iords': arr[0],
                        'mstar': arr[1],
                        't': np.repeat(t_val, len(arr[0])),
                        'z': np.repeat(z_val, len(arr[0])),
                        'type': np.repeat('accreted', len(arr[0])),
                    })
                    row.to_csv(filenames[k], mode='a', header=False)

        del DMOparticles
        print("Done with iteration", i)

    print(f'\nFinished. Wrote {n_instances} output files:')
    for fn in filenames:
        print(' ', fn)

    return filenames


def angmom_tag_multi_instance_hydro_dm(
    HYDROsim,
    n_instances,
    halonumber=1,
    free_param_value=0.01,
    output_prefix=None,
    mergers=True,
    track_cluster_file=None,
):
    '''
    Multi-instance angular momentum tagging on HYDRO simulation DM particles using DarkLight.

    Identical to angmom_tag_multi_instance but:
      - loads HYDRO snapshots and filters to .dm particles
      - calls DarkLight with DMO=False

    Inputs:
        HYDROsim         - tangos HYDRO simulation object
        n_instances      - number of independent DarkLight realisations (n=1 each)
        halonumber       - halo number (default 1)
        free_param_value - tagging fraction (default 0.01)
        output_prefix    - directory for output CSVs
        mergers          - whether to include accreted tagging (default True)

    Returns:
        list of output CSV filenames (length n_instances)
    '''

    DMOname = HYDROsim.path
    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(HYDROsim, halonumber)

    # Load track_cluster HDF5 if supplied — provides AHF halonums + cluster iords
    _tc_halonum_map   = None
    cluster_iords_map = None
    if track_cluster_file is not None:
        import h5py
        _tc_data = {}
        with h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_data[snap] = {
                        'halonum': int(f[snap]['main']['halonum'][()]),
                        'iords':   f[snap]['main']['iords'][:],
                    }
        _tc_halonum_map   = {s: d['halonum'] for s, d in _tc_data.items()}
        cluster_iords_map = {s: d['iords']   for s, d in _tc_data.items()}
        print(f'Loaded track_cluster file: {len(_tc_data)} snapshots, using AHF halonums + cluster iords')

    print(f'Running DarkLight (DMO=False) {n_instances} time(s) for main halo...')
    dl_histories = [
        DarkLight(main_halo, DMO=False, n=1, mergers=False)
        for _ in range(n_instances)
    ]

    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
    hmerge_added, z_set_vals = group_mergers(zmerge, hmerge)

    if len(red_all) != len(outputs):
        print('output array length does not match redshift and time arrays')

    if output_prefix is None:
        output_prefix = DMOname + '_hydrodm_tagged'
    os.makedirs(output_prefix, exist_ok=True)
    filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
    _header = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    for fn in filenames:
        _header.to_csv(fn, mode='w', header=True)

    def _mstar_at(mstar_arr, t_dl, t_target):
        idx = np.argmin(abs(t_dl - t_target))
        arr = np.asarray(mstar_arr)
        return float(np.mean(arr[:, idx] if arr.ndim == 2 else arr[idx]))

    for i in range(len(outputs)):
        gc.collect()
        print('Current snapshot -->', outputs[i])

        hDMO  = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        z_val = red_all[i]
        t_val = t_all[i]

        mass_selects_insitu = []
        for k in range(n_instances):
            t_dl, _, _, _, mstar_s_k, _ = dl_histories[k]
            msn = _mstar_at(mstar_s_k, t_dl, t_val)
            if msn == 0:
                mass_selects_insitu.append(0)
                continue
            msp = _mstar_at(mstar_s_k, t_dl, t_all[i - 1]) if i > 0 else 0.0
            mass_selects_insitu.append(int(msn - msp))

        merger_snap = (
            mergers and (i + 1 < len(red_all)) and (red_all[i + 1] in z_set_vals)
        )
        need_snap = any(m > 0 for m in mass_selects_insitu) or merger_snap
        if not need_snap:
            print("Done with iteration", i)
            continue

        _hydro_base = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')
        simfn = join(_hydro_base, DMOname, outputs[i])
        try:
            HYDROparticles = pynbody.load(simfn)
            HYDROparticles.physical_units()
        except Exception as e:
            print(f'--> failed to load snapshot: {e}, skipping')
            continue

        if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
        else:
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        if any(m > 0 for m in mass_selects_insitu):
            try:
                hDMO['r200c']
            except Exception:
                print("Couldn't load R200 at timestep:", i)
                del HYDROparticles
                continue

            if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
                h = HYDROparticles.halos(halo_numbers='v1')[int(_tc_halonum_map[outputs[i]])]
            else:
                h = HYDROparticles.halos()[int(halonums[i]) - 1]
            pynbody.analysis.halo.center(h)

            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(
                    h.d, overden=200, r_max=None, rho_def='critical')
            except Exception:
                print('could not calculate R200c')
                del HYDROparticles
                continue

            # Filter to DM only within r200
            dm_within = HYDROparticles.dm[
                sqrt(HYDROparticles.dm['pos'][:, 0] ** 2
                     + HYDROparticles.dm['pos'][:, 1] ** 2
                     + HYDROparticles.dm['pos'][:, 2] ** 2) <= r200c_pyn
            ]

            if cluster_iords_map is not None and outputs[i] in cluster_iords_map:
                dm_within = dm_within[np.isin(dm_within['iord'], cluster_iords_map[outputs[i]])]

            parts_sorted_angmom = rank_order_particles_by_angmom(dm_within)
            del dm_within

            if parts_sorted_angmom.shape[0] > 0:
                for k in range(n_instances):
                    if mass_selects_insitu[k] > 0:
                        arr = assign_stars_to_particles(
                            mass_selects_insitu[k], parts_sorted_angmom, float(free_param_value))
                        row = pd.DataFrame({
                            'iords': arr[0],
                            'mstar': arr[1],
                            't':    np.repeat(t_val, len(arr[0])),
                            'z':    np.repeat(z_val, len(arr[0])),
                            'type': np.repeat('insitu', len(arr[0])),
                        })
                        row.to_csv(filenames[k], mode='a', header=False)

        if merger_snap:
            t_id = int(np.where(z_set_vals == red_all[i + 1])[0][0])
            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                try:
                    h_merge = HYDROparticles.halos()[int(hDM.calculate('halo_number()')) - 1]
                    pynbody.analysis.halo.center(h_merge.dm)
                    r200c_acc = pynbody.analysis.halo.virial_radius(
                        h_merge.d, overden=200, r_max=None, rho_def='critical')
                except Exception as ex:
                    print('centering data unavailable, skipping', ex)
                    continue

                dm_acc = HYDROparticles.dm[
                    sqrt(HYDROparticles.dm['pos'][:, 0] ** 2
                         + HYDROparticles.dm['pos'][:, 1] ** 2
                         + HYDROparticles.dm['pos'][:, 2] ** 2) <= r200c_acc
                ]
                try:
                    acc_sorted = rank_order_particles_by_angmom(dm_acc)
                except Exception:
                    del dm_acc
                    continue
                del dm_acc

                for k in range(n_instances):
                    try:
                        _, _, _, _, _, mstar_merging_k = DarkLight(hDM, DMO=False, n=1, mergers=True)
                        if np.asarray(mstar_merging_k).size == 0:
                            continue
                    except Exception as e:
                        print(e, '-- skipping instance', k)
                        continue

                    mass_merge_k = float(np.asarray(mstar_merging_k).flat[-1])
                    if int(mass_merge_k) < 1:
                        continue

                    arr = assign_stars_to_particles(
                        int(mass_merge_k), acc_sorted, float(free_param_value))
                    row = pd.DataFrame({
                        'iords': arr[0],
                        'mstar': arr[1],
                        't':    np.repeat(t_val, len(arr[0])),
                        'z':    np.repeat(z_val, len(arr[0])),
                        'type': np.repeat('accreted', len(arr[0])),
                    })
                    row.to_csv(filenames[k], mode='a', header=False)

        del HYDROparticles
        print("Done with iteration", i)

    print(f'\nFinished. Wrote {n_instances} output files:')
    for fn in filenames:
        print(' ', fn)
    return filenames


def angmom_tag_multi_instance_hydro_mstars(
    HYDROsim,
    n_instances,
    halonumber=1,
    free_param_value=0.01,
    output_prefix=None,
    track_cluster_file=None,
):
    '''
    Multi-instance angular momentum tagging on HYDRO DM particles using the actual
    HYDRO simulation stellar mass (from SFR_histogram via integrate_sfr).

    No DarkLight — stellar mass is deterministic from the simulation. Running
    n_instances > 1 produces identical results but maintains a consistent output
    format with the DarkLight-based functions.

    Inputs:
        HYDROsim         - tangos HYDRO simulation object
        n_instances      - number of output instances (default 1 is sufficient)
        halonumber       - halo number (default 1)
        free_param_value - tagging fraction (default 0.01)
        output_prefix    - directory for output CSVs

    Returns:
        list of output CSV filenames (length n_instances)
    '''
    from .angular_momentum_tagging_HYDRO_DM import integrate_sfr

    DMOname = HYDROsim.path
    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(HYDROsim, halonumber)

    # Load track_cluster HDF5 if supplied — provides AHF halonums + cluster iords
    _tc_halonum_map   = None
    cluster_iords_map = None
    if track_cluster_file is not None:
        import h5py
        _tc_data = {}
        with h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_data[snap] = {
                        'halonum': int(f[snap]['main']['halonum'][()]),
                        'iords':   f[snap]['main']['iords'][:],
                    }
        _tc_halonum_map   = {s: d['halonum'] for s, d in _tc_data.items()}
        cluster_iords_map = {s: d['iords']   for s, d in _tc_data.items()}
        print(f'Loaded track_cluster file: {len(_tc_data)} snapshots, using AHF halonums + cluster iords')

    # Stellar mass history from HYDRO SFR histogram
    mstar_array, t_sfr = integrate_sfr(main_halo["SFR_histogram"], t_all[-1])
    print(f'Hydro mstar history: {len(mstar_array)} bins up to t={t_all[-1]:.2f} Gyr')

    def _mstar_at_hydro(t_target):
        idx = np.argmin(abs(t_sfr - t_target))
        return float(mstar_array[idx])

    if output_prefix is None:
        output_prefix = DMOname + '_hydromstars_tagged'
    os.makedirs(output_prefix, exist_ok=True)
    filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
    _header = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    for fn in filenames:
        _header.to_csv(fn, mode='w', header=True)

    for i in range(len(outputs)):
        gc.collect()
        print('Current snapshot -->', outputs[i])

        hDMO  = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        z_val = red_all[i]
        t_val = t_all[i]

        msn = _mstar_at_hydro(t_val)
        msp = _mstar_at_hydro(t_all[i - 1]) if i > 0 else 0.0
        mass_select = int(msn - msp)

        if mass_select <= 0:
            print("Done with iteration", i)
            continue

        try:
            hDMO['r200c']
        except Exception:
            print("Couldn't load R200 at timestep:", i)
            continue

        _hydro_base = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')
        simfn = join(_hydro_base, DMOname, outputs[i])
        try:
            HYDROparticles = pynbody.load(simfn)
            HYDROparticles.physical_units()
        except Exception as e:
            print(f'--> failed to load snapshot: {e}, skipping')
            continue

        if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
            h = HYDROparticles.halos(halo_numbers='v1')[int(_tc_halonum_map[outputs[i]])]
        else:
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            h = HYDROparticles.halos()[int(halonums[i]) - 1]
        pynbody.analysis.halo.center(h)

        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(
                h.d, overden=200, r_max=None, rho_def='critical')
        except Exception:
            print('could not calculate R200c')
            del HYDROparticles
            continue

        dm_within = HYDROparticles.dm[
            sqrt(HYDROparticles.dm['pos'][:, 0] ** 2
                 + HYDROparticles.dm['pos'][:, 1] ** 2
                 + HYDROparticles.dm['pos'][:, 2] ** 2) <= r200c_pyn
        ]

        if cluster_iords_map is not None and outputs[i] in cluster_iords_map:
            dm_within = dm_within[np.isin(dm_within['iord'], cluster_iords_map[outputs[i]])]

        parts_sorted_angmom = rank_order_particles_by_angmom(dm_within)
        del dm_within

        if parts_sorted_angmom.shape[0] > 0:
            arr = assign_stars_to_particles(mass_select, parts_sorted_angmom, float(free_param_value))
            for k in range(n_instances):
                row = pd.DataFrame({
                    'iords': arr[0],
                    'mstar': arr[1],
                    't':    np.repeat(t_val, len(arr[0])),
                    'z':    np.repeat(z_val, len(arr[0])),
                    'type': np.repeat('insitu', len(arr[0])),
                })
                row.to_csv(filenames[k], mode='a', header=False)

        del HYDROparticles
        print("Done with iteration", i)

    print(f'\nFinished. Wrote {n_instances} output files:')
    for fn in filenames:
        print(' ', fn)
    return filenames


def angmom_tag_dmo_hydro_mstars(
    DMOsim,
    HYDROsim,
    n_instances,
    halonumber=1,
    free_param_value=0.01,
    output_prefix=None,
    track_cluster_file=None,
):
    '''
    Multi-instance angular momentum tagging on DMO DM particles using stellar mass
    from the paired HYDRO simulation (SFR_histogram via integrate_sfr).

    Identical flow to angmom_tag_multi_instance_hydro_mstars but loads DMO snapshots
    instead of HYDRO snapshots. Stellar mass increments are sourced from HYDROsim's
    tangos SFR_histogram, so no DarkLight stochasticity is involved.

    Inputs:
        DMOsim           - tangos DMO simulation object (particle loading + r200c)
        HYDROsim         - tangos HYDRO simulation object (SFR_histogram only)
        n_instances      - number of output instances (results are identical; n=1 is fine)
        halonumber       - halo number in DMO sim (default 1)
        free_param_value - tagging fraction (default 0.01)
        output_prefix    - directory for output CSVs
        track_cluster_file - track_cluster HDF5; switches to AHF + filters by cluster iords

    Returns:
        list of output CSV filenames (length n_instances)
    '''
    from .angular_momentum_tagging_HYDRO_DM import integrate_sfr

    DMOname = DMOsim.path

    t_all, red_all, main_halo_dmo, halonums, outputs = load_indexing_data(DMOsim, halonumber)

    # Stellar mass history from paired HYDRO sim
    _, _, main_halo_hydro, _, _ = load_indexing_data(HYDROsim, halonumber)
    mstar_array, t_sfr = integrate_sfr(main_halo_hydro["SFR_histogram"], t_all[-1])
    print(f'Hydro mstar history: {len(mstar_array)} bins up to t={t_all[-1]:.2f} Gyr')

    def _mstar_at_hydro(t_target):
        idx = np.argmin(abs(t_sfr - t_target))
        return float(mstar_array[idx])

    # Load track_cluster HDF5 if supplied
    _tc_halonum_map   = None
    cluster_iords_map = None
    if track_cluster_file is not None:
        import h5py
        _tc_data = {}
        with h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_data[snap] = {
                        'halonum': int(f[snap]['main']['halonum'][()]),
                        'iords':   f[snap]['main']['iords'][:],
                    }
        _tc_halonum_map   = {s: d['halonum'] for s, d in _tc_data.items()}
        cluster_iords_map = {s: d['iords']   for s, d in _tc_data.items()}
        print(f'Loaded track_cluster file: {len(_tc_data)} snapshots, using AHF halonums + cluster iords')

    if output_prefix is None:
        output_prefix = DMOname + '_tagged_dmo_hydromstars'
    os.makedirs(output_prefix, exist_ok=True)
    filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
    _header = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    for fn in filenames:
        _header.to_csv(fn, mode='w', header=True)

    for i in range(len(outputs)):
        gc.collect()
        print('Current snapshot -->', outputs[i])

        hDMO  = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        z_val = red_all[i]
        t_val = t_all[i]

        msn = _mstar_at_hydro(t_val)
        msp = _mstar_at_hydro(t_all[i - 1]) if i > 0 else 0.0
        mass_select = int(msn - msp)

        if mass_select <= 0:
            print("Done with iteration", i)
            continue

        try:
            hDMO['r200c']
        except Exception:
            print("Couldn't load R200 at timestep:", i)
            continue

        simfn = join(config.get_path("pynbody_path"), DMOname, outputs[i])
        try:
            DMOparticles = pynbody.load(simfn)
            DMOparticles.physical_units()
        except Exception as e:
            print(f'--> failed to load snapshot: {e}, skipping')
            continue

        if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
            h = DMOparticles.halos(halo_numbers='v1')[int(_tc_halonum_map[outputs[i]])]
        else:
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            h = DMOparticles.halos()[int(halonums[i]) - 1]
        pynbody.analysis.halo.center(h)

        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(
                h.d, overden=200, r_max=None, rho_def='critical')
        except Exception:
            print('could not calculate R200c')
            del DMOparticles
            continue

        dm_within = DMOparticles.dm[
            sqrt(DMOparticles.dm['pos'][:, 0] ** 2
                 + DMOparticles.dm['pos'][:, 1] ** 2
                 + DMOparticles.dm['pos'][:, 2] ** 2) <= r200c_pyn
        ]

        if cluster_iords_map is not None and outputs[i] in cluster_iords_map:
            dm_within = dm_within[np.isin(dm_within['iord'], cluster_iords_map[outputs[i]])]

        parts_sorted_angmom = rank_order_particles_by_angmom(dm_within)
        del dm_within

        if parts_sorted_angmom.shape[0] > 0:
            arr = assign_stars_to_particles(mass_select, parts_sorted_angmom, float(free_param_value))
            for k in range(n_instances):
                row = pd.DataFrame({
                    'iords': arr[0],
                    'mstar': arr[1],
                    't':    np.repeat(t_val, len(arr[0])),
                    'z':    np.repeat(z_val, len(arr[0])),
                    'type': np.repeat('insitu', len(arr[0])),
                })
                row.to_csv(filenames[k], mode='a', header=False)

        del DMOparticles
        print("Done with iteration", i)

    print(f'\nFinished. Wrote {n_instances} output files:')
    for fn in filenames:
        print(' ', fn)
    return filenames


def angmom_tag_from_cluster_tree(
    DMOsim,
    cluster_tree_file,
    n_instances,
    halonumber=1,
    free_param_value=0.01,
    output_prefix=None,
    mergers=True,
):
    '''
    Multi-instance angular momentum tagging driven by a voxel cluster tree HDF5 file
    (produced by scripts/track_cluster.py).

    Instead of walking the tangos merger tree recursively, the cluster tree file encodes
    which DM particles belong to the main halo and each satellite at every snapshot.
    Tagging is a single forward pass:
      - Main branch iords → insitu tagging (DarkLight mass increment per snap)
      - Satellite branches → accreted tagging at the merger snapshot (DarkLight total mass)

    Inputs:
        DMOsim           - tangos simulation object
        cluster_tree_file - path to HDF5 produced by track_cluster.py
        n_instances      - number of independent DarkLight realisations (n=1 each)
        halonumber       - tangos halo number for main halo (default 1)
        free_param_value - tagging fraction (default 0.01)
        output_prefix    - directory for output CSVs (default: <sim_name>_tagged_tree)
        mergers          - whether to tag satellite branches (default True)

    Returns:
        list of output CSV filenames (length n_instances)
    '''
    import h5py

    DMOname = DMOsim.path

    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(DMOsim, halonumber)

    # t lookup by output name
    t_by_output   = {outputs[i]: t_all[i]   for i in range(len(outputs))}
    z_by_output   = {outputs[i]: red_all[i] for i in range(len(outputs))}

    # ── Load cluster tree ──────────────────────────────────────────────────────
    with h5py.File(cluster_tree_file, 'r') as f:
        tree_outputs = sorted(f.keys())   # all snapshots in tree (ascending)
        # per branch: {snap: iords}
        branch_iords = {}   # branch_id -> {output: np.array of iords}
        for snap in tree_outputs:
            for branch_id in f[snap].keys():
                if branch_id not in branch_iords:
                    branch_iords[branch_id] = {}
                branch_iords[branch_id][snap] = f[snap][branch_id]['iords'][:]

        # satellite merger snapshots: last snap where each sat exists (going forwards)
        sat_merger_snaps = {}
        sat_merger_halonums = {}
        for branch_id in branch_iords:
            if branch_id == 'main':
                continue
            sat_snaps = sorted(branch_iords[branch_id].keys())
            merger_snap = sat_snaps[-1]
            sat_merger_snaps[branch_id]   = merger_snap
            sat_merger_halonums[branch_id] = int(f[merger_snap][branch_id]['halonum'][()])

    print(f'Cluster tree: {len(tree_outputs)} snapshots, '
          f'{len(branch_iords)-1} satellite branch(es)')

    # ── DarkLight histories ────────────────────────────────────────────────────
    print(f'Running DarkLight for main halo ({n_instances} instances)...')
    dl_main = [DarkLight(main_halo, DMO=True, n=1, mergers=False) for _ in range(n_instances)]

    # DarkLight for each satellite (called at merger snapshot)
    dl_sat = {}   # branch_id -> list of DarkLight results (length n_instances)
    if mergers:
        for branch_id, merger_snap in sat_merger_snaps.items():
            halonum_sat = sat_merger_halonums[branch_id]
            try:
                hDM_sat = tangos.get_halo(
                    DMOname + '/' + merger_snap + '/halo_' + str(halonum_sat))
                print(f'Running DarkLight for {branch_id} (merger snap {merger_snap}, '
                      f'halo {halonum_sat})...')
                dl_sat[branch_id] = [
                    DarkLight(hDM_sat, DMO=True, n=1, mergers=True)
                    for _ in range(n_instances)
                ]
            except Exception as e:
                print(f'  DarkLight failed for {branch_id}: {e}, skipping')

    # ── Initialise output files ────────────────────────────────────────────────
    if output_prefix is None:
        output_prefix = DMOname + '_tagged_tree'
    os.makedirs(output_prefix, exist_ok=True)
    filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
    _header = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    for fn in filenames:
        _header.to_csv(fn, mode='w', header=True)

    def _mstar_at(mstar_arr, t_dl, t_target):
        idx = np.argmin(abs(t_dl - t_target))
        arr = np.asarray(mstar_arr)
        return float(np.mean(arr[:, idx] if arr.ndim == 2 else arr[idx]))

    # ── Main forward loop ──────────────────────────────────────────────────────
    for i, output in enumerate(outputs):
        gc.collect()
        print('Current snapshot -->', output)

        t_val = t_all[i]
        z_val = red_all[i]

        # ── Insitu mass selects ────────────────────────────────────────────────
        mass_selects_insitu = []
        for k in range(n_instances):
            t_dl, _, _, _, mstar_s_k, _ = dl_main[k]
            msn = _mstar_at(mstar_s_k, t_dl, t_val)
            if msn == 0:
                mass_selects_insitu.append(0)
                continue
            msp = _mstar_at(mstar_s_k, t_dl, t_all[i - 1]) if i > 0 else 0.0
            mass_selects_insitu.append(int(msn - msp))

        # satellites merging at this snapshot
        merging_now = [
            bid for bid, ms in sat_merger_snaps.items()
            if ms == output and bid in dl_sat
        ] if mergers else []

        need_snap = any(m > 0 for m in mass_selects_insitu) or len(merging_now) > 0
        if not need_snap:
            print("Done with iteration", i)
            continue

        # ── Load snapshot ONCE ────────────────────────────────────────────────
        simfn = join(config.get_path("pynbody_path"), DMOname, output)
        try:
            DMOparticles = pynbody.load(simfn)
            DMOparticles.physical_units()
        except Exception as e:
            print(f'--> failed to load snapshot: {e}, skipping')
            continue

        pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        # ── Insitu block ──────────────────────────────────────────────────────
        if any(m > 0 for m in mass_selects_insitu) and output in branch_iords.get('main', {}):
            main_cluster_iords = branch_iords['main'][output]

            try:
                h = pynbody.halo.hop.HOPCatalogue(DMOparticles)[int(halonums[i]) - 1]
                pynbody.analysis.halo.center(h)
                r200c_pyn = pynbody.analysis.halo.virial_radius(
                    h.d, overden=200, r_max=None, rho_def='critical')
            except Exception as e:
                print(f'  Centering failed: {e}, skipping insitu')
                del DMOparticles
                print("Done with iteration", i)
                continue

            DMOparts_insitu = DMOparticles[
                sqrt(DMOparticles['pos'][:, 0] ** 2
                     + DMOparticles['pos'][:, 1] ** 2
                     + DMOparticles['pos'][:, 2] ** 2) <= r200c_pyn
            ]
            DMOparts_insitu = DMOparts_insitu[
                np.isin(DMOparts_insitu['iord'], main_cluster_iords)
            ]

            parts_sorted_angmom = rank_order_particles_by_angmom(DMOparts_insitu)
            del DMOparts_insitu

            if parts_sorted_angmom.shape[0] > 0:
                for k in range(n_instances):
                    if mass_selects_insitu[k] > 0:
                        arr = assign_stars_to_particles(
                            mass_selects_insitu[k], parts_sorted_angmom, float(free_param_value))
                        row = pd.DataFrame({
                            'iords': arr[0],
                            'mstar': arr[1],
                            't':    np.repeat(t_val, len(arr[0])),
                            'z':    np.repeat(z_val, len(arr[0])),
                            'type': np.repeat('insitu', len(arr[0])),
                        })
                        row.to_csv(filenames[k], mode='a', header=False)

        # ── Accreted block ────────────────────────────────────────────────────
        for branch_id in merging_now:
            sat_cluster_iords = branch_iords[branch_id][output]
            halonum_sat = sat_merger_halonums[branch_id]

            try:
                h_sat = pynbody.halo.hop.HOPCatalogue(DMOparticles)[int(halonum_sat) - 1]
                pynbody.analysis.halo.center(h_sat.dm)
                r200c_sat = pynbody.analysis.halo.virial_radius(
                    h_sat.d, overden=200, r_max=None, rho_def='critical')
            except Exception as e:
                print(f'  Centering failed for {branch_id}: {e}, skipping')
                continue

            DMOparts_acc = DMOparticles[
                sqrt(DMOparticles['pos'][:, 0] ** 2
                     + DMOparticles['pos'][:, 1] ** 2
                     + DMOparticles['pos'][:, 2] ** 2) <= r200c_sat
            ]
            DMOparts_acc = DMOparts_acc[
                np.isin(DMOparts_acc['iord'], sat_cluster_iords)
            ]

            try:
                acc_sorted = rank_order_particles_by_angmom(DMOparts_acc)
            except Exception as e:
                print(f'  angmom ranking failed for {branch_id}: {e}, skipping')
                del DMOparts_acc
                continue
            del DMOparts_acc

            for k in range(n_instances):
                try:
                    _, _, _, _, _, mstar_merging_k = dl_sat[branch_id][k]
                    if np.asarray(mstar_merging_k).size == 0:
                        continue
                except Exception as e:
                    print(f'  DarkLight result unavailable for {branch_id} instance {k}: {e}')
                    continue

                mass_merge_k = float(np.asarray(mstar_merging_k).flat[-1])
                if int(mass_merge_k) < 1:
                    continue

                arr = assign_stars_to_particles(
                    int(mass_merge_k), acc_sorted, float(free_param_value))
                row = pd.DataFrame({
                    'iords': arr[0],
                    'mstar': arr[1],
                    't':    np.repeat(t_val, len(arr[0])),
                    'z':    np.repeat(z_val, len(arr[0])),
                    'type': np.repeat('accreted', len(arr[0])),
                })
                row.to_csv(filenames[k], mode='a', header=False)

        del DMOparticles
        print("Done with iteration", i)

    print(f'\nFinished. Wrote {n_instances} output files:')
    for fn in filenames:
        print(' ', fn)

    return filenames


def angmom_tag_over_full_sim_recursive(DMOsim,tstep, halonumber, free_param_value = 0.001,free_param_value_acc = None ,pynbody_path  = None, particle_storage_filename=None, AHF_centers_filepath=None, mergers = True, df_tagged_particles=None ,tag_typ='insitu',acc_halo_path_tagged=None,main_halo_paths=None):

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
    #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

    # extracts name of DMO simulation
    DMOname = DMOsim.path

    t_all,red_all,main_halo,halonums,outputs = load_indexing_data(DMOsim,halonumber)
    
    print(config.get("darklight","n"))

    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,n=config.get("darklight","n"),mergers=False) 
    
    mstar_s_insitu = np.asarray(mstar_s_insitu[0])
    print("t,z:",t,redshift)

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
    AHF_centers = pd.read_csv(os.path.join(AHF_centers_filepath,str(DMOname)+"_rec.csv")) if type(AHF_centers_filepath) != type(None) else None
    AHF_centers_acc = pd.read_csv(os.path.join(AHF_centers_filepath,str(DMOname)+"_accreted_rec.csv")) if type(AHF_centers_filepath) != type(None) else None
    
    tagged_iords_to_write = np.array([])
    tagged_types_to_write = np.array([])
    tagged_mstars_to_write = np.array([])
    
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    
    # record of tagged objects for the recursive run where the loop goes through all merging objects 
    acc_halo_path_tagged = np.array([]) if (type(acc_halo_path_tagged) == type(None)) else acc_halo_path_tagged 
    
    if  type(df_tagged_particles) == type(None):    
        df_tagged_particles = pd.DataFrame({'iords':tagged_iords_to_write, 'mstar':tagged_mstars_to_write,'t':ts_to_write,'z':zs_to_write,'type':tagged_types_to_write})
    
    #if (particle_storage_filename != None): 
    #    df_tagged_particles.to_csv(particle_storage_filename)
    
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
        msn = float(np.mean(np.asarray(mstar_s_insitu)[:, idrz] if np.asarray(mstar_s_insitu).ndim == 2 else np.asarray(mstar_s_insitu)[idrz]))              
        
        print("msn:",msn)

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap idrz_previous==None and msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            elif idrz_previous >= 0:
                msp = float(np.mean(np.asarray(mstar_s_insitu)[:, idrz_previous] if np.asarray(mstar_s_insitu).ndim == 2 else np.asarray(mstar_s_insitu)[idrz_previous]))
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

                simfn = join(config.get_path("pynbody_path"),DMOname,outputs[i])
                
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
                
                #AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]
                
                if (tag_typ == "insitu") :

                    AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]

                if (tag_typ != "insitu"):

                    AHF_halonum_acc = AHF_centers_acc[AHF_centers_acc["snapshot"] == outputs[i]] if type(AHF_centers_filepath) != type(None) else None
                    HOP_halonum_acc = int(halonums[i])
                    AHF_halonum_accreted = AHF_halonum_acc[AHF_halonum_acc["HOP halonum"] == HOP_halonum_acc]["AHF halonum"].values[0]

                    AHF_crossref = AHF_halonum_accreted



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
            #pynbody.analysis.angmom.faceon(h.dm[h.dm['r']<5])
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
        
            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
            
            DMOparticles_insitu_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] 
        
            DMOparticles_insitu_only = DMOparticles_insitu_only.dm

            #uncomment to remove subhalos from tagging insitu 
            ####DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]
            
            particles_sorted_by_angmom = rank_order_particles_by_angmom( DMOparticles_insitu_only)
            
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
            if particle_storage_filename != None:
                row_to_write.to_csv(particle_storage_filename+"_"+str(output[i])+".csv")

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
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,mergers=True,n=config.get("darklight","n"))

                    mstar_merging = mstar_merging[0]

                    #occupation=occupation_frac, pre_method='fiducial_with_turnover', post_scatter_method='flat',DMO=True,mergers = True
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
                halonumber_hDM = hDM.calculate_for_progenitors('halo_number()')[0][0]
                print('halonum merging:',halonumber_hDM)
                

                if type(main_halo_paths) != type(None): 
                    
                    if ( len(np.where(np.isin(main_halo_paths,acc_halo_path_tagged) == True)[0]) != 0):
                        continue

                # if halo has not been tagged on before, we want to perform tagging over its full lifetime (upto the current snap)
                if ( len(np.where(np.isin(acc_halo_path,acc_halo_path_tagged) == True)[0]) == 0 ):
                    
                    
                    print('---recursion triggered -----')
                    

                    df_tagged_particles,acc_halo_path_tagged = angmom_tag_over_full_sim_recursive(DMOsim,tidx,halonumber_hDM, free_param_value = float(free_param_value_acc),free_param_value_acc = float(free_param_value_acc),pynbody_path = pynbody_path, df_tagged_particles=df_tagged_particles,tag_typ='accreted',AHF_centers_filepath=AHF_centers_filepath,acc_halo_path_tagged=acc_halo_path_tagged)
                    
                    
        
                    print('---recursion end -----')
                                
                    
                else:
                    
                    if len(mstar_merging)==0:
                        continue
    
                    mass_select_merge= mstar_merging[-1] - mstar_merging[-2]  if len(mstar_merging) > 1 else mstar_merging[-1]
    
                    
                    if int(mass_select_merge)<1:
                        continue
                    
                    
                    
                    simfn = join(config.get_path("pynbody_path"),DMOname,outputs[i])

                    if ((float(mass_select_merge) >0) and (decision2==True)):
                        # try to load in the data from this snapshot
                        
                        if (type(AHF_centers_filepath) != type(None)):
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
        
                        DMOparticles_acc_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn_acc] 
                                                    
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
                        if particle_storage_filename != None: 
                            row_to_write_acc.to_csv(particle_storage_filename+"_"+str(output[i])+".csv",mode='a',header=False) 
                        print('writing accreted particles to output file')
              
                        del DMOparticles_acc_only
                  
                            
        if decision==True or decl==True:
            del DMOparticles
    
    
        print("Done with iteration",i)
        
            
    return df_tagged_particles,acc_halo_path_tagged


def angmom_tag_multi_instance_recursive(
    DMOsim,
    n_instances,
    tstep,
    halonumber,
    free_param_value=0.001,
    free_param_value_acc=None,
    pynbody_path=None,
    output_prefix=None,
    AHF_centers_filepath=None,
    mergers=True,
    dfs_tagged_particles=None,
    tag_typ='insitu',
    acc_halo_path_tagged=None,
    main_halo_paths=None,
    filenames=None,
    cluster_file=None,
    cluster_iords_map=None,
):
    '''
    Multi-instance variant of angmom_tag_over_full_sim_recursive.

    Runs n_instances independent DarkLight realisations (each with n=1) over the
    full simulation, loading each snapshot once and writing one output CSV per instance.

    At the top level call, pass only DMOsim, n_instances, tstep, halonumber and any
    optional kwargs.  filenames and dfs_tagged_particles are initialised internally and
    threaded through recursive calls automatically.

    Returns: (dfs_tagged_particles, acc_halo_path_tagged)
        dfs_tagged_particles - list of N DataFrames (one per instance)
        filenames are written incrementally to disk
    '''

    if free_param_value_acc is None:
        free_param_value_acc = free_param_value

    DMOname = DMOsim.path

    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(DMOsim, halonumber)

    # Load cluster iords lookup from HDF5 if supplied (top-level call only; passed through on recursion)
    if cluster_iords_map is None and cluster_file is not None:
        import h5py
        with h5py.File(cluster_file, 'r') as f:
            cluster_iords_map = {k: f[k][:] for k in f.keys()}
        print(f'Loaded cluster iords for {len(cluster_iords_map)} snapshots from {cluster_file}')

    # N independent DarkLight histories for the main halo (n=1 each for genuine stochasticity)
    print(f'Running DarkLight {n_instances} time(s) for halo {halonumber} (n=1 each)...')
    dl_histories = [
        DarkLight(main_halo, DMO=True, n=1, mergers=False)
        for _ in range(n_instances)
    ]
    # Extract 1D mstar arrays — dl[4][0] mirrors the [0] indexing in the original
    t_arrays     = [dl[0] for dl in dl_histories]
    mstar_arrays = [np.asarray(dl[4][0]) for dl in dl_histories]

    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)

    if len(red_all) != len(outputs):
        print('output array length does not match redshift and time arrays')

    hmerge_added, z_set_vals = group_mergers(zmerge, hmerge)

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids   = np.array([])

    AHF_centers     = pd.read_csv(os.path.join(AHF_centers_filepath, str(DMOname) + "_rec.csv"))         if AHF_centers_filepath is not None else None
    AHF_centers_acc = pd.read_csv(os.path.join(AHF_centers_filepath, str(DMOname) + "_accreted_rec.csv")) if AHF_centers_filepath is not None else None

    # Initialise N DataFrames and N output files (top-level call only)
    _empty = pd.DataFrame({'iords': [], 'mstar': [], 't': [], 'z': [], 'type': []})
    if dfs_tagged_particles is None:
        dfs_tagged_particles = [_empty.copy() for _ in range(n_instances)]

    if filenames is None:
        if output_prefix is None:
            output_prefix = DMOname + '_tagged_recursive'
        os.makedirs(output_prefix, exist_ok=True)
        filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(n_instances)]
        for fn in filenames:
            _empty.to_csv(fn, mode='w', header=True)

    acc_halo_path_tagged = np.array([]) if acc_halo_path_tagged is None else acc_halo_path_tagged

    if len(acc_halo_path_tagged) > 0:
        halo_path = main_halo.calculate_for_progenitors('path()')
        print("halopath:", halo_path)
        main_halo_paths = np.array([])
        main_halo_paths = np.append(main_halo_paths, halo_path[0][0])

        if len(np.where(np.isin(main_halo_paths, acc_halo_path_tagged) == True)[0]) > 0:
            print("overlap at :", main_halo_paths[np.where(np.isin(main_halo_paths, acc_halo_path_tagged) == True)])
            print("for halo :", acc_halo_path_tagged)
            return dfs_tagged_particles, acc_halo_path_tagged

    halo_path = main_halo.calculate_for_progenitors('path()')
    acc_halo_path_tagged = np.append(acc_halo_path_tagged, halo_path[0][0])

    # Main snapshot loop
    for i in range(len(outputs)):
        gc.collect()

        if any(len(t_arrays[k]) == 0 for k in range(n_instances)):
            continue

        decision  = False
        decision2 = False
        decl      = False

        print('Current snapshot -->', outputs[i])

        hDMO  = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        z_val = red_all[i]
        t_val = t_all[i]

        # Compute per-instance mass_select (cheap, no I/O)
        mass_selects = []
        for k in range(n_instances):
            idrz      = np.argmin(abs(t_arrays[k] - t_val))
            idrz_prev = np.argmin(abs(t_arrays[k] - t_all[i - 1])) if idrz > 0 else None
            msn = float(mstar_arrays[k][idrz])
            if msn == 0:
                mass_selects.append(0)
                continue
            msp = float(mstar_arrays[k][idrz_prev]) if idrz_prev is not None else 0.0
            mass_selects.append(int(msn - msp))

        if not any(m > 0 for m in mass_selects):
            # Still check mergers below — but don't load the snap yet
            pass

        # Load snapshot ONCE if any instance needs it or a merger snap is coming
        merger_snap = (
            mergers
            and (i + 1 < len(red_all))
            and (red_all[i + 1] in z_set_vals)
        )
        need_snap = any(m > 0 for m in mass_selects) or merger_snap

        if need_snap:
            if AHF_centers_filepath is not None:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]

            simfn = join(config.get_path("pynbody_path"), DMOname, outputs[i])
            try:
                print(simfn)
                print('loading in DMO particles')
                DMOparticles = pynbody.load(simfn)
                DMOparticles.physical_units()
                print('loaded data')
                decision = True
            except Exception as e:
                print(e)
                print('--> failed to load snapshot, skipping')
                print("Done with iteration", i)
                continue

        # Insitu block
        if any(m > 0 for m in mass_selects):
            try:
                hDMO['r200c']
            except Exception:
                print("Couldn't load R200 at timestep:", i)
                del DMOparticles
                print("Done with iteration", i)
                continue

            subhalo_iords = np.array([])

            if AHF_centers_filepath is None:
                print("Halonum:", int(halonums[i]) - 1)
                h = DMOparticles.halos()[int(halonums[i]) - 1]
            else:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                if tag_typ == 'insitu':
                    AHF_crossref = AHF_centers[AHF_centers['snapshot'] == outputs[i]]['AHF halonum'].values[0]
                else:
                    AHF_halonum_acc  = AHF_centers_acc[AHF_centers_acc['snapshot'] == outputs[i]]
                    HOP_halonum_acc  = int(halonums[i])
                    AHF_crossref     = AHF_halonum_acc[AHF_halonum_acc['HOP halonum'] == HOP_halonum_acc]['AHF halonum'].values[0]

                h = DMOparticles.halos(halo_numbers="v1")[int(AHF_crossref)]
                children_ahf_int = h.properties['children']
                halo_catalogue = DMOparticles.halos(halo_numbers="v1")
                for ch in children_ahf_int:
                    if ch != AHF_crossref:
                        subhalo_iords = np.append(subhalo_iords, halo_catalogue[int(ch)].dm['iord'])
                h = h[np.logical_not(np.isin(h['iord'], subhalo_iords))] if len(subhalo_iords) > 0 else h

            pynbody.analysis.halo.center(h.dm)

            try:
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')
            except Exception:
                print('could not calculate R200c')
                del DMOparticles
                print("Done with iteration", i)
                continue

            DMOparts_insitu = DMOparticles[
                sqrt(DMOparticles['pos'][:, 0] ** 2
                     + DMOparticles['pos'][:, 1] ** 2
                     + DMOparticles['pos'][:, 2] ** 2) <= r200c_pyn
            ].dm

            if cluster_iords_map is not None and outputs[i] in cluster_iords_map:
                DMOparts_insitu = DMOparts_insitu[
                    np.isin(DMOparts_insitu['iord'], cluster_iords_map[outputs[i]])
                ]

            # Angular-momentum ranking ONCE per snap
            parts_sorted_angmom = rank_order_particles_by_angmom(DMOparts_insitu)
            del DMOparts_insitu

            # Fan across N instances
            if parts_sorted_angmom.shape[0] > 0:
                for k in range(n_instances):
                    if mass_selects[k] <= 0:
                        continue
                    arr = assign_stars_to_particles(mass_selects[k], parts_sorted_angmom, float(free_param_value))
                    row = pd.DataFrame({
                        'iords': arr[0],
                        'mstar': arr[1],
                        't':    np.repeat(t_val,  len(arr[0])),
                        'z':    np.repeat(z_val,  len(arr[0])),
                        'type': np.repeat(tag_typ, len(arr[0])),
                    })
                    dfs_tagged_particles[k] = pd.concat([dfs_tagged_particles[k], row], ignore_index=True)
                    if filenames[k] is not None:
                        row.to_csv(filenames[k], mode='a', header=False)

        # Mergers block
        if merger_snap:
            decision2 = False if decision else True
            decl = False
            t_id = int(np.where(z_set_vals == red_all[i + 1])[0][0])

            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:', hDM)

                try:
                    prob_occupied = calculate_poccupied(hDM, 2.5e7)
                except Exception as e:
                    print(e)
                    print("poccupied couldn't be calculated")
                    continue

                if np.random.random() > prob_occupied:
                    print('Skipped')
                    continue

                tidx             = int(np.where(np.asarray(DMOsim.timesteps[:]) == hDMO.timestep)[0][0])
                acc_halo_path    = hDM.calculate_for_progenitors('path()')
                halonumber_hDM   = hDM.calculate_for_progenitors('halo_number()')[0][0]
                print('halonum merging:', halonumber_hDM)

                if main_halo_paths is not None:
                    if len(np.where(np.isin(main_halo_paths, acc_halo_path_tagged) == True)[0]) != 0:
                        continue

                if len(np.where(np.isin(acc_halo_path, acc_halo_path_tagged) == True)[0]) == 0:
                    # Halo not yet tagged — recurse for all N instances
                    print('---recursion triggered -----')
                    dfs_tagged_particles, acc_halo_path_tagged = angmom_tag_multi_instance_recursive(
                        DMOsim,
                        n_instances,
                        tidx,
                        halonumber_hDM,
                        free_param_value=float(free_param_value_acc),
                        free_param_value_acc=float(free_param_value_acc),
                        pynbody_path=pynbody_path,
                        AHF_centers_filepath=AHF_centers_filepath,
                        dfs_tagged_particles=dfs_tagged_particles,
                        tag_typ='accreted',
                        acc_halo_path_tagged=acc_halo_path_tagged,
                        filenames=filenames,
                        cluster_iords_map=cluster_iords_map,
                    )
                    print('---recursion end -----')

                else:
                    # Halo already tagged — apply remaining mass at this snap, per instance
                    # Load snap if not already loaded
                    if decision2:
                        simfn = join(config.get_path("pynbody_path"), DMOname, outputs[i])
                        if AHF_centers_filepath is not None:
                            pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                        try:
                            DMOparticles = pynbody.load(simfn)
                            DMOparticles.physical_units()
                            print('loaded data in mergers')
                        except Exception:
                            print('--> DMO particle data exists but failed to read it, skipping!')
                            continue
                        decision2 = False
                        decl = True

                    # Center and rank accreted particles ONCE for this merger halo
                    try:
                        HOP_halonum_acc = int(hDM.calculate('halo_number()'))
                        if AHF_centers_filepath is not None:
                            AHF_halonum_acc      = AHF_centers_acc[AHF_centers_acc['snapshot'] == outputs[i]]
                            AHF_halonum_accreted = AHF_halonum_acc[AHF_halonum_acc['HOP halonum'] == HOP_halonum_acc]['AHF halonum'].values[0]
                            h_merge = DMOparticles.halos(halo_numbers="v1")[AHF_halonum_accreted]
                        else:
                            h_merge = DMOparticles.halos()[HOP_halonum_acc - 1]
                        pynbody.analysis.halo.center(h_merge, mode='hyb')
                        r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                    except Exception as ex:
                        print('centering data unavailable, skipping', ex)
                        continue

                    DMOparts_acc = DMOparticles[
                        sqrt(DMOparticles['pos'][:, 0] ** 2
                             + DMOparticles['pos'][:, 1] ** 2
                             + DMOparticles['pos'][:, 2] ** 2) <= r200c_pyn_acc
                    ]
                    try:
                        acc_sorted = rank_order_particles_by_angmom(DMOparts_acc.dm)
                    except Exception:
                        del DMOparts_acc
                        continue
                    del DMOparts_acc

                    # Each instance gets its own DarkLight draw (n=1) for the remaining mass
                    for k in range(n_instances):
                        try:
                            _, _, _, _, _, mstar_merging_k = DarkLight(hDM, DMO=True, n=1, mergers=True)
                            mstar_merging_k = mstar_merging_k[0]
                        except Exception as e:
                            print(e, '-- skipping instance', k)
                            continue

                        if np.asarray(mstar_merging_k).size == 0:
                            continue

                        _m_arr = np.asarray(mstar_merging_k)
                        mass_merge_k = float(_m_arr.flat[-1] - _m_arr.flat[-2]) if _m_arr.size > 1 else float(_m_arr.flat[-1])
                        if int(mass_merge_k) < 1:
                            continue

                        arr = assign_stars_to_particles(int(mass_merge_k), acc_sorted, float(free_param_value_acc))
                        row = pd.DataFrame({
                            'iords': arr[0],
                            'mstar': arr[1],
                            't':    np.repeat(t_val,     len(arr[0])),
                            'z':    np.repeat(z_val,     len(arr[0])),
                            'type': np.repeat('accreted', len(arr[0])),
                        })
                        dfs_tagged_particles[k] = pd.concat([dfs_tagged_particles[k], row], ignore_index=True)
                        if filenames[k] is not None:
                            row.to_csv(filenames[k], mode='a', header=False)

        if decision or decl:
            del DMOparticles

        print("Done with iteration", i)

    return dfs_tagged_particles, acc_halo_path_tagged



