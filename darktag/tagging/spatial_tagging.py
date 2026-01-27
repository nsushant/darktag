
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
import random
from ...config import config




def rhalf2D_dm(particles):
    '''
    
    Calculates radius that encloses half the given particles.

    Assumes each particle positions have been centered on main halo.  Adopts
    same 'luminosity' for each particle.  Creates list of projected distances
    from halo center (as seen from the z-axis), sorts this to get the distances
    in increasing order, then choses the distance that encloses half the particles.
    
    '''
    rproj = np.sqrt(particles['x']**2 + particles['y']**2)
    rproj.sort()
    if round(len(particles)/2)>0:
        return rproj[ round(len(particles)/2) ]
    else:
        return rproj
                    

def get_mass(m,a,r1,r2):

    '''
    
    calculates the mass enclosed at distances r1 and r2 
    from the center of the main halo 
    according to the plummer profile 

    '''
    
    x1 = m*(r1**3)/((r1**2+a**2)**(3.0/2.0))
    x2 = m*(r2**3)/((r2**2+a**2)**(3.0/2.0))
    return x2-x1



def plum_const(hDMO,z_val,insitu,r200):
    
    '''
    
    calculates the value of the Plummer constant based on 
    the given simulation redshift and halo virial radius. 
    
    '''
    
    if insitu == 'insitu':
        return ((0.015*r200)/1.3) if z_val > 4 else ((10**(0.1*r200 - 4.2))/1.3)
    else:
        return ((0.015*r200)/1.3)

#prod_binned_df(z_val, msn,mass_select,chosen_parts,DMOparticles,hDMO,'insitu',a_coeff,r200)


def prod_binned_df(z_val, mstar, mass_select, chosen_parts, DMOparticles, hDMO,insitu,a_coeff,r200):

    '''
    
    Log bins particles located at radial distances of < 10 * Plummer radius according to their spatial positions. 
    
    '''

    print('this is how many',len(DMOparticles))
    
    # calculate 'a' to construct plummer profile 
    if insitu == 'insitu':
        a = plum_const(hDMO,z_val,'insitu',r200)
    elif insitu == 'accreted':
        a = plum_const(hDMO,z_val,'accreted',r200)
    

    print('appended to, a_coeff')
    #store 'a' for this snap 
    #a_coeff = np.append(a_coeff,a)
            
    # filter out particles outside the plummer tidal radius 'a'.
    particles_in_selection_radius = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= 15*a ]

    selection_mass = 1112

    iords_of_parts = list(particles_in_selection_radius['iord'])
    positions_of_parts = particles_in_selection_radius['pos']

    if chosen_parts.shape[0] >0:

        print('removal of duplicates took place')
        
        ignore_these = [iords_of_parts.index(i) for i in iords_of_parts if i in chosen_parts]
        
        iords_of_parts = np.delete(iords_of_parts,ignore_these)
        
        positions_of_parts = np.delete(positions_of_parts,ignore_these,axis=0)
    

    # get 3d pos vec magnitude array 
    R_dists = get_dist(positions_of_parts)

    num_of_bins=5      
    #generate log-spaced bins 
    bins = np.logspace(np.log10(min(R_dists)),np.log10(max(R_dists)),num=num_of_bins,base=10)
    bins = np.insert(bins,0,0)
        
    # bin data (assigns bin numbers to data)
    binning = np.digitize(R_dists,bins,right=True)

    # put elements that don't fit anywhere, into the last bin 
    binning = np.where(binning==len(bins),len(bins)-1,binning)
            
    # generate left edges 
    bz_prev = binning-1
    bz_prev = np.where(bz_prev<0,0,bz_prev)


    binned_df = pd.DataFrame({'bin_right':bins[binning],'bin_left':bins[bz_prev], 'iords':iords_of_parts,'pos':R_dists, 'pos_x': positions_of_parts[:,0], 'pos_y': positions_of_parts[:,1],'pos_z': positions_of_parts[:,2]})

    binned_df = binned_df.drop_duplicates(subset=['iords'],keep='first')

    return binned_df, bins, a, a_coeff, selection_mass


def get_bins(bins, binned_df, M_0, a, a_coeff, msp, red_all, t_all, i, insitu,sm):

    '''
    
    Given a binned dataframe (containing particles binned by position), the function 
    associates stellar mass with dm particles by using the Plummer Profile 
    to calculate the stellar mass growth at each binned distance. 
    
    '''

    ch_parts_2 = np.array([])
    out_num = np.array([])
    tgyr_of_choice = np.array([])

    r_of_choice = np.array([])
    p_typ = np.array([])
    a_storage = np.array([])
    m_storage = np.array([])
    for bx in bins[1:]:

        idx_b = np.array(np.where(bins==bx))
        b_left = np.array(idx_b) - 1
                
        #b = binned_df['iords'][binned_df['bin_right'] == bx][binned_df['bin_left'] == bins[b_left[0][0]]].reset_index()
        #po = binned_df[binned_df['bin_right']==bx][binned_df['bin_left']==bins[b_left[0][0]]][['pos_x','pos_y','pos_z']].reset_index()
        df_bin = binned_df[(binned_df['bin_right'] == bx) & (binned_df['bin_left'] == bins[b_left[0][0]])].reset_index()
        po = df_bin[['pos_x','pos_y','pos_z']]
        b = df_bin['iords']
        
        #chaged!!!!!!!
        if insitu == 'insitu':
            print('length of a_coeff after insitu engaged', len(a_coeff))
            if len(a_coeff)>1:
                mass_binned = get_mass(M_0,a,bins[b_left[0][0]],bx)
                
            else:
                print('a_coeff < 0 at this timestep')
                mass_binned = get_mass(M_0,a,bins[b_left[0][0]],bx)

        if insitu == 'accreted':
            print('accreted tagging engaged')
            mass_binned = get_mass(M_0,a,bins[np.array(idx_b) - 1],bx)

        if (mass_binned/1112)<=0:
            print('Decrease/ Zero!! mass_binned = ', mass_binned)
            continue
                
        print('binned mass',mass_binned)
        #print(sm)
        s = int(mass_binned/sm)+(1 if np.random.random() < (mass_binned % sm)/sm else 0)
        #print('This is how many particles-------------------------------------------------------------', int(mass_binned/sm),s,len(b.index))
                
        if s <= len(list(b.index)):
            print('route 1 ;  s < b.index length')
            print('Stellar mass in bin:', s*sm)
            choose_parts = random.sample(list(b.index),s)
            print('choose_out_of_this',b.index)
            print('these were chosen',choose_parts)
                    
        elif len(list(b.index)) < s :
            print('route 2 ; s-1 > b.index length')
            choose_parts = list(b.index)
            print(choose_parts)

        else:
            continue 
                       
        if len(choose_parts) != len(set(choose_parts)):
            print('unequality in choice', len(choose_parts),len(set(choose_parts)))
        else:
            print('All Unique iords are found ')
        
        if len(choose_parts)>0:
            ch_parts_2 = np.append(ch_parts_2,(b.iloc[choose_parts].values))
    
            out_num = np.append(out_num, np.repeat(i,len(choose_parts)))
            r_of_choice = np.append(r_of_choice, np.repeat(red_all[i],len(choose_parts)))
            tgyr_of_choice = np.append(tgyr_of_choice,np.repeat(t_all[i],len(choose_parts)))

            p_typ = np.append(p_typ, np.repeat(insitu,len(choose_parts)))

            a_storage = np.append(a_storage, np.repeat(a,len(choose_parts)))
            m_storage = np.append(m_storage,np.repeat(M_0,len(choose_parts)))
            

    return ch_parts_2,out_num,tgyr_of_choice,r_of_choice,p_typ,a_storage,m_storage


def spatial_tag_over_full_sim(DMOsim, pynbody_path  = None, occupation_frac = 'all', particle_storage_filename=None, mergers=True):
    
    # Use config path if pynbody_path is not provided
    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")

    '''

    Given a tangos simulation, the function tags particles with stellar mass based on the Plummer profile in each snapshot. 
    In doing so it 'grows' the tagged stellar population over the full simulation. 

    '''
    
    # keeps count of the number of mergers
    mergers_count = 0
    simname = DMOsim.path
    
    t_all, red_all, main_halo,halonums,outputs = load_indexing_data(DMOsim,1)
    
    # iterating over all the simulations in the 'sims' list
   
    #darklight stellar masses used for the selection of insitu particles
    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers=False,poccupied=occupation_frac)
    
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
        hDMO = tangos.get_halo(join(config.get_path("tangos_path"), outputs[i], 'halo_'+str(halonums[i])))

        # value of redshift at the current timestep
        z_val = red_all[i]
        t_val = t_all[i]
        
        # round each value in the redhsifts list from DarkLight to 6 decimal places
        np_round_to_6 = np.round(np.array(abs(redshift)), 6)
      
        # generate path to snapshot 
        simfn = join(pynbody_path, outputs[i])

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
                    
                    prob_occupied = calculate_poccupied(hDM,occupation_frac)
                    print('successfully calculated poccupied')
                    
                else:
                    print("poccupied couldn't be calculated --> dm mass profile unavailable in tangos db")
                    continue
                
                if (np.random.random() > prob_occupied):
                    print('skipped in accordance with occupation fraction selected')
                    continue
                                                                                                                                                                        
                try:
                    # loading in the properties of the halo from darklight as above
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,mergers=True,poccupied=occupation_frac)
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

    if particle_storage_filename != None: 
        df_spatially_tagged_particles.to_csv(particle_storage_filename)
    
    return df_spatially_tagged_particles


