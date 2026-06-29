from .spatial_tagging import *
from .angular_momentum_tagging import *
from ..config import config
from .clustering import cluster_tagged_particles
from ..analysis.calculate import calc_3D_cm, produce_lums_grouped, calc_halflight


def _voxel_pick_cluster(positions, iords, voxel_size, prev_iords=None):
    """
    Voxel-based spatial clustering on a set of particles.

    Snaps each particle to a 3-D grid, then builds connected components via
    26-neighbour union-find on occupied voxels.  Returns a boolean mask
    selecting the best cluster.

    Parameters
    ----------
    positions  : (N, 3) float array – particle positions in physical units (kpc)
    iords      : (N,)  int array   – particle Iord values
    voxel_size : float             – voxel edge length (same units as positions)
    prev_iords : array-like or None – iords from previous snapshot;
                                      if given, pick cluster with most overlap;
                                      if None, pick largest cluster

    Returns
    -------
    mask : (N,) bool array, or None if no particles / no cluster found
    """
    if len(positions) == 0:
        return None

    vx = np.floor(positions[:, 0] / voxel_size).astype(np.int64)
    vy = np.floor(positions[:, 1] / voxel_size).astype(np.int64)
    vz = np.floor(positions[:, 2] / voxel_size).astype(np.int64)

    # voxel key → list of particle indices
    voxel_dict = {}
    for idx in range(len(iords)):
        key = (vx[idx], vy[idx], vz[idx])
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(idx)

    # Union-Find with path compression
    parent = {k: k for k in voxel_dict}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    key_set = set(voxel_dict)
    for (x, y, z) in list(voxel_dict):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nb = (x + dx, y + dy, z + dz)
                    if nb in key_set:
                        union((x, y, z), nb)

    # group particle indices by cluster root
    clusters = {}
    for key, idxs in voxel_dict.items():
        root = find(key)
        if root not in clusters:
            clusters[root] = []
        clusters[root].extend(idxs)

    if not clusters:
        return None

    if prev_iords is not None and len(prev_iords) > 0:
        prev_set = set(prev_iords)
        best_root = max(clusters,
                        key=lambda r: sum(1 for idx in clusters[r] if iords[idx] in prev_set))
    else:
        best_root = max(clusters, key=lambda r: len(clusters[r]))

    mask = np.zeros(len(iords), dtype=bool)
    for idx in clusters[best_root]:
        mask[idx] = True
    return mask

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

    elif tagging_method == "angular momentum recursive":

        df_tagged,l = angmom_tag_over_full_sim_recursive(DMO_database, -1, halonumber, free_param_value = free_param_val, pynbody_path  = path_to_particle_data )

    elif tagging_method == 'spatial' : 
        
        df_tagged = spatial_tag_over_full_sim(DMO_database, pynbody_path  = path_to_particle_data, occupation_frac = 'all', particle_storage_filename=None, mergers= include_mergers)
    
    return df_tagged



def calculate_reffs_over_full_sim(DMOsim, particles_tagged,  pynbody_path  = None, path_AHF_halonums=None, from_file = False ,from_dataframe=False,save_to_file=True,AHF_centers_supplied=False,machine='astro',physics='edge1',halo_number=0, reffs_fname='reffs.csv', use_clustering=True, use_ahf=False):
    
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
    PrevBGMMIords = np.array([])

    AHF_centers = pd.read_csv(str(path_AHF_halonums)) if AHF_centers_supplied == True else None
            
    for i in range(len(outputs))[::-1]:

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
        except Exception as e:
            print(f'--> DMO particle data exists but failed to read it, skipping! Error: {e}')
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
                    if use_ahf:
                        h = DMOparticles.halos(halo_numbers='v1')[int(halonums[i]) - 1]
                    else:
                        hop_cat = pynbody.halo.hop.HOPCatalogue(DMOparticles)
                        h = hop_cat[int(halonums[i]) - 1]


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
                
            
            if use_ahf or AHF_halonums is not None:
                halo_cat = DMOparticles.halos(halo_numbers='v1')
            else:
                halo_cat = hop_cat
            children_dm, children_st, sub_halonums = get_child_iords(h, halo_cat, DMO_state='DMO')
            
            DMOparticles.physical_units()
            pynbody.analysis.halo.center(h)
            pynbody.analysis.angmom.faceon(h.dm)

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
        
        if use_clustering and len(particle_selection_reff_tot) > 0:
            clustering_cfg = config.get('tagging', 'clustering')
            masses_for_clustering = np.array([
                data_grouped.loc[iord]['mstar']
                for iord in particle_selection_reff_tot['iord']
            ])
            labels, best_label, _ = cluster_tagged_particles(
                particles=particle_selection_reff_tot,
                prev_iords=PrevBGMMIords if len(PrevBGMMIords) > 0 else None,
                method=clustering_cfg.get('method', 'dbscan'),
                feature_cols=clustering_cfg.get('features', ['x', 'y']),
                scale=clustering_cfg.get('scale', False),
                sample_weight=masses_for_clustering,
                eps=config.get_with_default('dbscan', 'eps', 0.05),
                dbscan_min_samples=config.get_with_default('dbscan', 'min_samples', 2),
                min_cluster_size=config.get_with_default('hdbscan', 'min_cluster_size', 10),
                hdbscan_min_samples=config.get_with_default('hdbscan', 'min_samples', None),
                cluster_selection_epsilon=config.get_with_default('hdbscan', 'cluster_selection_epsilon', 0.0),
                cluster_selection_method=config.get_with_default('hdbscan', 'cluster_selection_method', 'eom'),
                allow_single_cluster=config.get_with_default('hdbscan', 'allow_single_cluster', True),
                max_cluster_size=config.get_with_default('hdbscan', 'max_cluster_size', None),
            )
            if best_label == -1:
                continue
            particle_selection_reff_tot = particle_selection_reff_tot[np.where(labels == best_label)]
            PrevBGMMIords = np.delete(PrevBGMMIords, np.arange(len(PrevBGMMIords)))
            PrevBGMMIords = np.append(PrevBGMMIords, np.asarray(particle_selection_reff_tot['iord']))

        #print('m200 value---->',hDMO['M200c'])
        
        if (len(particle_selection_reff_tot))==0:
            print('skipped!')
            continue
        else:
    
            masses = [ data_grouped.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

            if len(particles_only_insitu) > 0:
                masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                cen_stars = calc_3D_cm(particles_only_insitu, masses_insitu)
            else:
                print('no insitu particles at this snap, centering on all tagged particles')
                cen_stars = calc_3D_cm(particle_selection_reff_tot, masses)

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
            hlight_r = calc_halflight(particle_selection_reff_tot, lum_for_each_part, band='v', cylindrical=True)
            
            print(hlight_r)
            
            lum_based_halflight = np.append(lum_based_halflight,hlight_r)
            
            stored_reff_z = np.append(stored_reff_z,red_all[i])
            stored_time = np.append(stored_time, t_all[i])
               
            stored_reff = np.append(stored_reff,float(R_half))
            try:
                kravtsov = hDMO['r200c']*0.02
            except KeyError:
                print('r200c not available for this halo, storing NaN for kravtsov')
                kravtsov = float('nan')
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


def calculate_reffs_multi_instance(
    DMOsim,
    tagged_dir,
    pynbody_path=None,
    path_AHF_halonums=None,
    AHF_centers_supplied=False,
    halo_number=0,
    output_dir=None,
    save_to_file=True,
    use_clustering=True,
    use_ahf=False,
    voxel_fraction=0.05,
):
    '''
    Multi-instance variant of calculate_reffs_over_full_sim.

    Loads each snapshot ONCE and calculates reffs for all instance particle files
    found in tagged_dir (instance_000.csv, instance_001.csv, ...).
    Each instance maintains its own PrevVoxelIords for independent voxel-cluster tracking.

    Inputs:
        DMOsim            - tangos simulation object
        tagged_dir        - directory containing instance_*.csv tagged particle files
        pynbody_path      - path to snapshot data (defaults to config)
        path_AHF_halonums - path to AHF halo number crossref CSV (optional)
        AHF_centers_supplied - whether AHF centering file is provided
        halo_number       - halo number (default 0)
        output_dir        - directory to write output reff CSVs (default: tagged_dir + '_reffs')
        save_to_file      - whether to write CSVs incrementally (default True)
        voxel_fraction    - voxel size as fraction of r200c (default 0.05)

    Returns:
        list of reff DataFrames, one per instance
    '''

    if pynbody_path is None:
        pynbody_path = config.get_path("pynbody_path")

    if output_dir is None:
        output_dir = tagged_dir.rstrip('/') + '_reffs'
    os.makedirs(output_dir, exist_ok=True)

    # Discover instance files
    instance_files = sorted([
        f for f in os.listdir(tagged_dir) if f.startswith('instance_') and f.endswith('.csv')
    ])
    if len(instance_files) == 0:
        raise FileNotFoundError(f'No instance_*.csv files found in {tagged_dir}')
    n_instances = len(instance_files)
    print(f'Found {n_instances} instance files in {tagged_dir}')

    # AHF setup
    AHF_halonums = None
    if path_AHF_halonums is None:
        pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    else:
        if os.path.isfile(path_AHF_halonums):
            AHF_halonums = pd.read_csv(path_AHF_halonums)
            if len(AHF_halonums['snapshot']) > 0:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
            else:
                pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
        else:
            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

    AHF_centers = pd.read_csv(str(path_AHF_halonums)) if AHF_centers_supplied else None

    # Tangos indexing (shared across all instances)
    simname = DMOsim.path
    split = simname.split('_')
    halonum = split[0][4:]
    DMOname = 'Halo' + halonum + '_DMO' + ('' if len(split) == 2 else ('_' + '_'.join(split[2:])))

    main_halo = DMOsim.timesteps[-1].halos[int(halo_number)]
    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
    t_all    = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all  = main_halo.calculate_for_progenitors('z()')[0][::-1]

    outputs_all   = np.array([DMOsim.timesteps[i].__dict__['extension']  for i in range(len(DMOsim.timesteps))])
    times_tangos  = np.array([DMOsim.timesteps[i].__dict__['time_gyr']   for i in range(len(DMOsim.timesteps))])
    outputs = outputs_all[np.isin(times_tangos, t_all)]
    outputs.sort()

    if len(red_all) != len(outputs):
        print('output array length does not match redshift and time arrays')

    # Load all instance particle DataFrames
    all_data = [pd.read_csv(os.path.join(tagged_dir, f)) for f in instance_files]
    all_data_t = [np.asarray(d['t'].values) for d in all_data]

    # Output filenames
    out_fnames = [os.path.join(output_dir, f.replace('instance_', 'reff_instance_')) for f in instance_files]

    # Per-instance state — pre-populated from existing output CSVs if resuming
    PrevVoxelIords  = [np.array([]) for _ in range(n_instances)]
    stored_reff    = [np.array([]) for _ in range(n_instances)]
    stored_reff_z  = [np.array([]) for _ in range(n_instances)]
    stored_time    = [np.array([]) for _ in range(n_instances)]
    kravtsov_r     = [np.array([]) for _ in range(n_instances)]
    lum_halflight  = [np.array([]) for _ in range(n_instances)]
    processed_outputs = [set() for _ in range(n_instances)]

    # build a t → output_name map for reverse-lookup on resume
    t_to_output = {round(float(t_all[i]), 8): outputs[i] for i in range(len(outputs))}

    for k, fname in enumerate(out_fnames):
        if os.path.isfile(fname):
            try:
                existing = pd.read_csv(fname, index_col=0)
                if len(existing) > 0:
                    stored_reff[k]   = existing['reff'].values
                    stored_reff_z[k] = existing['z'].values
                    stored_time[k]   = existing['t'].values
                    kravtsov_r[k]    = existing['kravtsov'].values
                    lum_halflight[k] = existing['halflight'].values
                    # map stored t values back to output names (string keys, no float ambiguity)
                    for tv in existing['t'].values:
                        out_name = t_to_output.get(round(float(tv), 8))
                        if out_name is not None:
                            processed_outputs[k].add(out_name)
                    print(f'  instance {k:03d}: resuming, {len(existing)} snapshots already done')
            except Exception as e:
                print(f'  instance {k:03d}: could not read existing output ({e}), starting fresh')

    # ── Main snapshot loop (reversed so voxel clustering seeds from z=0) ───────
    for i in range(len(outputs))[::-1]:
        gc.collect()

        # Skip entire snapshot if all instances already have it
        if all(outputs[i] in processed_outputs[k] for k in range(n_instances)):
            print(f'Skipping {outputs[i]} (all instances done)')
            continue

        print('Current snapshot -->', outputs[i])

        # ── Snap-level work (done ONCE) ───────────────────────────────────────
        hDMO   = tangos.get_halo(DMOname + '/' + outputs[i] + '/halo_' + str(halonums[i]))
        t_val  = t_all[i]
        z_val  = red_all[i]

        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue if AHF_halonums is not None else pynbody.halo.hop.HOPCatalogue]

        simfn = join(pynbody_path, outputs[i])
        try:
            DMOparticles = pynbody.load(simfn)
        except Exception as e:
            print(f'--> Failed to load snapshot, skipping! Error: {e}')
            continue

        try:
            if not AHF_centers_supplied:
                if AHF_halonums is not None:
                    halonum_snap = AHF_halonums[AHF_halonums["snapshot"] == str(outputs[i])]["AHF halonum"].values
                    h = DMOparticles.halos(halo_numbers='v1')[int(halonum_snap)]
                else:
                    if use_ahf:
                        h = DMOparticles.halos(halo_numbers='v1')[int(halonums[i]) - 1]
                    else:
                        hop_cat = pynbody.halo.hop.HOPCatalogue(DMOparticles)
                        h = hop_cat[int(halonums[i]) - 1]
            else:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                h = DMOparticles.halos()[int(AHF_crossref)]

            if use_ahf or AHF_halonums is not None:
                halo_cat = DMOparticles.halos(halo_numbers='v1')
            else:
                halo_cat = hop_cat
            children_dm, children_st, sub_halonums = get_child_iords(h, halo_cat, DMO_state='DMO')
            DMOparticles.physical_units()
            pynbody.analysis.halo.center(h)
            pynbody.analysis.angmom.faceon(h.dm)
        except Exception as e:
            print('centering data unavailable', e)
            continue

        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')
        except Exception:
            print('could not calculate R200c')
            continue

        try:
            kravtsov = hDMO['r200c'] * 0.02
        except KeyError:
            print('r200c not available for this halo, storing NaN for kravtsov')
            kravtsov = float('nan')

        # Filter snap particles once
        within_r200 = sqrt(DMOparticles['pos'][:, 0]**2 +
                           DMOparticles['pos'][:, 1]**2 +
                           DMOparticles['pos'][:, 2]**2) <= r200c_pyn
        DMOparts_snap = DMOparticles[within_r200]
        DMOparts_snap = DMOparts_snap[np.logical_not(np.isin(DMOparts_snap['iord'], children_dm))]

        # ── Per-instance work ─────────────────────────────────────────────────
        for k in range(n_instances):
            if len(np.where(all_data_t[k] <= float(t_val))) == 0:
                continue

            dt_all_k      = all_data[k][all_data[k]['t'] <= t_val]
            data_grouped_k = dt_all_k.groupby(['iords']).sum()
            selected_iords_tot_k = data_grouped_k.index.values

            if selected_iords_tot_k.shape[0] == 0:
                continue

            if outputs[i] in processed_outputs[k]:
                continue

            data_insitu_k = dt_all_k[dt_all_k['type'] == 'insitu'].groupby(['iords']).sum()
            selected_iords_insitu_k = data_insitu_k.index.values

            particle_sel_k   = DMOparts_snap[np.isin(DMOparts_snap['iord'], selected_iords_tot_k)]    if len(selected_iords_tot_k) > 0     else []
            parts_insitu_k   = DMOparts_snap[np.isin(DMOparts_snap['iord'], selected_iords_insitu_k)] if len(selected_iords_insitu_k) > 0  else []

            if len(particle_sel_k) == 0:
                continue

            # Voxel clustering with this instance's own PrevVoxelIords
            if use_clustering:
                pos_k    = np.array(particle_sel_k['pos'])
                iords_k  = np.asarray(particle_sel_k['iord'])
                vox_size = float(r200c_pyn) * voxel_fraction
                mask_k   = _voxel_pick_cluster(
                    pos_k, iords_k, vox_size,
                    prev_iords=PrevVoxelIords[k] if len(PrevVoxelIords[k]) > 0 else None,
                )
                if mask_k is None or mask_k.sum() == 0:
                    continue
                particle_sel_k  = particle_sel_k[mask_k]
                PrevVoxelIords[k] = np.asarray(particle_sel_k['iord'])

            masses_k = [data_grouped_k.loc[n]['mstar'] for n in particle_sel_k['iord']]

            if len(parts_insitu_k) > 0:
                masses_insitu_k = [data_insitu_k.loc[iord]['mstar'] for iord in parts_insitu_k['iord']]
                cen_stars_k = calc_3D_cm(parts_insitu_k, masses_insitu_k)
            else:
                cen_stars_k = calc_3D_cm(particle_sel_k, masses_k)

            particle_sel_k['pos'] -= cen_stars_k

            distances_k = np.sqrt(particle_sel_k['x']**2 + particle_sel_k['y']**2)
            idxs_sorted = np.argsort(distances_k)
            sorted_dists = np.sort(distances_k)
            dist_ordered_iords = np.asarray(particle_sel_k['iord'][idxs_sorted])

            sorted_masses_k = [data_grouped_k.loc[n]['mstar'] for n in dist_ordered_iords]
            cumsum_k = np.cumsum(sorted_masses_k)
            R_half_k = sorted_dists[np.where(cumsum_k >= (cumsum_k[-1] / 2))[0][0]]

            lum_k = produce_lums_grouped(dt_all_k, particle_sel_k['iord'], t_val)
            hlight_k = calc_halflight(particle_sel_k, lum_k, band='v', cylindrical=True)

            particle_sel_k['pos'] += cen_stars_k

            stored_reff[k]   = np.append(stored_reff[k],   float(R_half_k))
            stored_reff_z[k] = np.append(stored_reff_z[k], z_val)
            stored_time[k]   = np.append(stored_time[k],   t_val)
            kravtsov_r[k]    = np.append(kravtsov_r[k],    kravtsov)
            lum_halflight[k] = np.append(lum_halflight[k], hlight_k)

            if save_to_file:
                df_k = pd.DataFrame({
                    'halflight': lum_halflight[k],
                    'reff':      stored_reff[k],
                    'z':         stored_reff_z[k],
                    't':         stored_time[k],
                    'kravtsov':  kravtsov_r[k],
                })
                df_k.to_csv(out_fnames[k])

        del DMOparticles, DMOparts_snap

    # Build final DataFrames
    dfs_reff = []
    for k in range(n_instances):
        df_k = pd.DataFrame({
            'halflight': lum_halflight[k],
            'reff':      stored_reff[k],
            'z':         stored_reff_z[k],
            't':         stored_time[k],
            'kravtsov':  kravtsov_r[k],
        })
        dfs_reff.append(df_k)
        if save_to_file:
            df_k.to_csv(out_fnames[k])
            print(f'Wrote {out_fnames[k]}')

    return dfs_reff


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
        except Exception as e:
            print(f'--> DMO particle data exists but failed to read it, skipping! Error: {e}')
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

            if len(particles_only_insitu) > 0:
                masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                cen_stars = calc_3D_cm(particles_only_insitu, masses_insitu)
            else:
                print('no insitu particles at this snap, centering on all tagged particles')
                cen_stars = calc_3D_cm(particle_selection_reff_tot, masses)

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
            try:
                kravtsov = hDMO['r200c']*0.02
            except KeyError:
                print('r200c not available for this halo, storing NaN for kravtsov')
                kravtsov = float('nan')
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

    
