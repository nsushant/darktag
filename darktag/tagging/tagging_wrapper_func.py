from .spatial_tagging import *
from .angular_momentum_tagging import *
from ...config import config

def get_child_iords(halo,halo_catalog,DMO_state='fiducial'):

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

                dm_2nd_gen,st_2nd_gen,sub_halonums_2nd_gen = get_child_iords(halo_catalog[child],halo_catalog,DMO_state)

                children_dm = np.append(children_dm,dm_2nd_gen)
                children_st = np.append(children_st,st_2nd_gen)
                sub_halonums = np.append(sub_halonums,sub_halonums_2nd_gen)
            #else:                                                                                                                                                                                                                                             
            #    print("there were no star or dark-matter iord arrays")                                                                                                                                                                                        

    #else:                                                                                                                                                                                                                                                     
    #    print("did not find children in halo properties list")                                                                                                                                                                                                

    return children_dm,children_st,sub_halonums



def tag_particles(DMO_database, path_to_particle_data = None, tagging_method = 'angular momentum', free_param_val = 0.01, include_mergers = True, halonumber = 1):

    # Use config path if path_to_particle_data not provided
    if path_to_particle_data is None:
        path_to_particle_data = config.get_path("pynbody_path")

    if tagging_method == 'angular momentum':
        
        df_tagged = angmom_tag_over_full_sim(DMO_database, halonumber, free_param_value = free_param_val, pynbody_path  = path_to_particle_data, mergers = include_mergers)

    if tagging_method == "angular momentum recursive":

        df_tagged,l = angmom_tag_over_full_sim_recursive(DMO_database, -1, halonumber, free_param_value = free_param_val, pynbody_path  = path_to_particle_data )

    if tagging_method == 'spatial' : 
        
        df_tagged = spatial_tag_over_full_sim(DMO_database, pynbody_path  = path_to_particle_data, occupation_frac = 'all', particle_storage_filename=None, mergers= include_mergers)
    
    return df_tagged



def calculate_reffs_over_full_sim(DMOsim, particles_tagged,  pynbody_path  = None, path_AHF_halonums=None, from_file = False ,from_dataframe=False,save_to_file=True,AHF_centers_supplied=False,machine='astro',physics='edge1',halo_number=0):
    
    # Use config path if pynbody_path not provided
    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")
    #used paths
    
    AHF_halonums = None

    if type(path_AHF_halonums) == type(None): 
        print("Proceeding with HOP catalogue")
        pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

    else:
        if os.path.isfile(path_AHF_halonums): 
    
            AHF_halonums = pd.read_csv(path_AHF_halonums) 
    
            if len(AHF_halonums['snapshot']) > 0:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]            
    
            else:
                print("AHF halonums file at "+path_AHF_halonums+" is empty, using HOP catalogue")
                pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
        else: 
            print("AHF halonumsfile at"+path_AHF_halonums+" does not exist, using HOP catalogue")
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]


    simname = DMOsim.path
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
            exit()
    elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'

    if simname[-3] == 'x':
        DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

    else:
        DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:]))) #if split[1]=='fiducial' else None
        
    # get particle data at z=0 for DMO sims, if available
    if DMOname==None:
        print('--> DMO particle does not data exists, skipping!')
        exit()
                    
    main_halo = DMOsim.timesteps[-1].halos[int(halo_number)]
    
    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
   
    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all = main_halo.calculate_for_progenitors('z()')[0][::-1] 
    
    outputs_all = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])
    times_tangos = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

    outputs = outputs_all[np.isin(times_tangos, t_all)]
    
    outputs.sort()

    print(outputs)

    #load in the two files containing the particle data
    if ( len(red_all) != len(outputs) ) : 
        print('output array length does not match redshift and time arrays')

    data_particles = pd.read_csv(particles_tagged) if from_dataframe==False else particles_tagged

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
        
        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
        if type(AHF_halonums) == type(None):
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        #for  the given path,entry,snapshot at given index generate a string that includes them
        simfn = join(pynbody_path,outputs[i])
        
        # try to load in the data from this snapshot
        try:  DMOparticles = pynbody.load(simfn)

        # where this data isn't available, notify the user.
        except:
            print('--> DMO particle data exists but failed to read it, skipping!')
            continue
        
        # once the data from the snapshot has been loaded, .physical_units()
        # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
        #DMOparticles.physical_units()

        try:
            if AHF_centers_supplied==False:
                
                if type(AHF_halonums) != type(None):
                    print('halonums cat', DMOparticles.halos(halo_numbers='v1'),DMOparticles.halos(halo_numbers='v1').keys())
                    halonum_snap = AHF_halonums[AHF_halonums["snapshot"] == str(outputs[i])]["AHF halonum"].values
                    
                    h = DMOparticles.halos(halo_numbers='v1')[int(halonum_snap)]                        
                    
                else:
                    #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                    h = DMOparticles.halos(halo_numbers='v1')[int(halonums[i])-1]


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
                
            
            children_dm,children_st,sub_halonums = get_child_iords(h,DMOparticles.halos(halo_numbers='v1'),DMO_state='DMO')
            
            DMOparticles.physical_units()    
            pynbody.analysis.halo.center(h)

        except Exception as e:
            print('centering data unavailable',e)
            continue


        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')

        except:
            print('could not calculate R200c')
            continue
        
    

        DMOparticles = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ]        
        
        DMOparticles = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],children_dm))]

        particle_selection_reff_tot = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else []
        
        particles_only_insitu = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_insitu_only)] if len(selected_iords_insitu_only) > 0 else []
        

        #print('m200 value---->',hDMO['M200c'])
        
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


'''
def calculate_reffs_over_full_sim(DMOsim, data_particles_tagged, pynbody_path  = None , AHF_centers_file = None):



    Given a tangos simulation, the function performs angular momentum based tagging over the full simulation. 

    Inputs: 

    DMOsim - tangos simulation 
    pynbody_path - path to particle data 
    data_particles_tagged - dataframe containing tagged particle data (tagged mstar, particle IDs, tagging times)
    
    Returns: 
    
    dataframe with half-mass radii calculated using tagged particles. 
    


    
    
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                    
    sims = [str(sim_name)]

    DMOname = DMOsim.path
    
    t_all, red_all, main_halo,halonums,outputs = load_indexing_data(DMOsim,1)
    print(outputs)
    

    #load in the two files containing the particle data
    if ( len(red_all) != len(outputs) ) : 
        print('output array length does not match redshift and time arrays')
 

    data_t = np.asarray(data_particles_tagged['t'].values)
    
    stored_reff = np.array([])
    stored_reff_acc = np.array([])
    stored_reff_z = np.array([])
    stored_time = np.array([])
    kravtsov_r = np.array([])
    stored_reff_tot = np.array([])
    KE_energy = np.array([])
    PE_energy = np.array([])

    AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_supplied == True else None
            
    for i in range(len(outputs)):

        gc.collect()

        
        if len(np.where(data_t <= float(t_all[i]))) == 0:
            continue

        
        dt_all = data_particles_tagged[data_particles_tagged['t']<=t_all[i]]

        data_grouped = dt_all.groupby(['iords']).last()

        selected_iords_tot = data_grouped.index.values

        data_insitu = data_grouped[data_grouped['type'] == 'insitu']
        
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
        simfn = join(pynbody_path,outputs[i])
        
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
            if AHF_centers_file==None:
                h = DMOparticles.halos()[int(halonums[i])-1]
                
            elif AHF_centers_file != None:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                
                AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                    
                h = DMOparticles.halos()[int(AHF_crossref)] 
                        
                children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                        
                child_str_l = children_ahf[0][1:-1].split()

                children_ahf_int = list(map(float, child_str_l))

                    
                #pynbody.analysis.halo.center(h)
                    
                #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                
                
                halo_catalogue = DMOparticles.halos()
                
                subhalo_iords = np.array([])
                    
                for i in children_ahf_int:
                            
                    subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(i)].dm['iord'])
                                                                                                                                             
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
                

                
            pynbody.analysis.halo.center(h)
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

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

        
        if (len(particle_selection_reff_tot))==0:
            print('skipped!')
            continue
        else:

            dfnew = data_particles_tagged[data_particles_tagged['t']<=t_all[i]].groupby(['iords']).last()
    
            masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

            masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                
            cen_stars = calc_3D_cm(particles_only_insitu,masses_insitu)
            
            particle_selection_reff_tot['pos'] -= cen_stars
            
            masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

            #particle_selection_reff_tot['pos'] -= cen_stars 

            distances =  np.sqrt(particle_selection_reff_tot['x']**2 + particle_selection_reff_tot['y']**2 + particle_selection_reff_tot['z']**2)

            #caculate the center of mass using all the tagged particles
            #cen_of_mass = center_on_tagged(distances,masses)
            
                        
            idxs_distances_sorted = np.argsort(distances)

            sorted_distances = np.sort(distances)

            distance_ordered_iords = np.asarray(particle_selection_reff_tot['iord'][idxs_distances_sorted])
            
            print('array lengths',len(set(distance_ordered_iords)),len(distance_ordered_iords))

            sorted_massess = [dfnew.loc[n]['mstar'] for n in distance_ordered_iords]
            
            cumilative_sum = np.cumsum(sorted_massess)

            R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]
            #print(cumilative_sum)
            
            halfmass_radius = []

            stored_reff_z = np.append(stored_reff_z,red_all[i])
            stored_time = np.append(stored_time, t_all[i])
               
            stored_reff = np.append(stored_reff,float(R_half))
            kravtsov = hDMO['r200c']*0.02
            kravtsov_r = np.append(kravtsov_r,kravtsov)

            particle_selection_reff_tot['pos'] += cen_stars

            print('halfmass radius:',R_half)
            print('Kravtsov_radius:',kravtsov)
            
        
  

    print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

    df_reff = pd.DataFrame({'reff':stored_reff,'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
    
    
    return df_reff


'''
def calculate_rhalf(DMOsim, data_particles_tagged, pynbody_path  = None, path_AHF_halonums = None, from_dataframe=True,from_file=False): 
    
    # Use config path if pynbody_path not provided
    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")
        
    return calculate_reffs_over_full_sim( DMOsim, data_particles_tagged, pynbody_path  = pynbody_path , path_AHF_halonums = path_AHF_halonums,from_file=from_file, from_dataframe = from_dataframe)

    
