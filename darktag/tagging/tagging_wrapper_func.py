from .spatial_tagging import *
from .angular_momentum_tagging import *
from ..config import config
from .clustering import cluster_tagged_particles
from ..analysis.calculate import calc_3D_cm, produce_lums_grouped, calc_halflight


def _voxel_pick_cluster(positions, iords, voxel_size, prev_iords=None,
                        max_degree=20, size_jump=2.0):
    """
    Iterative voxel clustering using scipy.ndimage.label (C-level, no Python loops).

    Builds a 3D boolean grid from voxelised positions, then for degree=1..max_degree
    calls ndimage.label with an all-ones structuring element of size (2D+1)^3.
    Stops when the main cluster stabilises or a size jump signals satellite absorption.

    Parameters
    ----------
    positions   : (N, 3) float array – positions in kpc
    iords       : (N,)  int array
    voxel_size  : float – voxel edge in kpc (e.g. 0.08)
    prev_iords  : array-like or None – iords from previous snapshot for overlap tracking
    max_degree  : int   – maximum connectivity radius in voxel steps (default 20)
    size_jump   : float – ratio threshold that signals satellite absorption (default 2.0)

    Returns
    -------
    mask : (N,) bool array, or None if no particles found
    """
    from scipy.ndimage import label as ndimage_label, binary_dilation

    if len(positions) == 0:
        return None

    vx = np.floor(positions[:, 0] / voxel_size).astype(np.int64)
    vy = np.floor(positions[:, 1] / voxel_size).astype(np.int64)
    vz = np.floor(positions[:, 2] / voxel_size).astype(np.int64)

    # Shift to zero-based indices; int32 halves memory vs int64 for these index arrays
    ox, oy, oz = int(vx.min()), int(vy.min()), int(vz.min())
    gx = (vx - ox).astype(np.int32)
    gy = (vy - oy).astype(np.int32)
    gz = (vz - oz).astype(np.int32)
    del vx, vy, vz

    nx, ny, nz = int(gx.max()) + 1, int(gy.max()) + 1, int(gz.max()) + 1

    # Build boolean occupancy grid — fully vectorised, no Python loop
    grid = np.zeros((nx, ny, nz), dtype=bool)
    grid[gx, gy, gz] = True

    # np.isin is O(N log M) — replaces the previous O(N·M) Python loop
    if prev_iords is not None and len(prev_iords) > 0:
        prev_mask = np.isin(iords, prev_iords)
    else:
        prev_mask = None

    # Fixed (3,3,3) structure — ndimage.label only supports this size.
    # Degree-D connectivity is achieved by dilating the grid D times before labelling:
    # two original voxels merge iff their Chebyshev distance <= D.
    structure3 = np.ones((3, 3, 3), dtype=bool)

    best_mask = None
    prev_size = 0
    # Carry the dilated grid forward — each iteration adds exactly 1 dilation step
    # instead of restarting from scratch (degree=20 → 20 passes vs 210).
    expanded = grid.copy()

    for degree in range(1, max_degree + 1):
        expanded = binary_dilation(expanded, structure=structure3, iterations=1)
        labeled, n_clusters = ndimage_label(expanded, structure=structure3)

        if n_clusters == 0:
            del labeled
            break

        # Extract per-particle labels then immediately free the large labeled grid
        particle_labels = labeled[gx, gy, gz]
        del labeled

        # Count particles per label
        label_counts = np.bincount(particle_labels, minlength=n_clusters + 1)
        label_counts[0] = 0  # ignore background

        if prev_mask is not None:
            # Count prev-snapshot overlap per label — vectorised with bincount
            prev_labels = particle_labels[prev_mask]
            label_overlap = np.bincount(prev_labels, minlength=n_clusters + 1)
            label_overlap[0] = 0
            main_label = int(label_overlap.argmax()) if label_overlap.max() > 0 else int(label_counts.argmax())
        else:
            main_label = int(label_counts.argmax())

        curr_size = int(label_counts[main_label])

        if prev_size > 0 and curr_size >= prev_size * size_jump:
            print(f'    voxel iter: degree {degree} caused size jump '
                  f'{prev_size}→{curr_size} (×{curr_size/prev_size:.1f}), '
                  f'stopping at degree {degree - 1}')
            break

        # Build particle mask for this label
        curr_mask = particle_labels == main_label
        best_mask = curr_mask

        if curr_size == prev_size:
            print(f'    voxel iter: stabilised at degree {degree} '
                  f'with {curr_size} particles')
            break

        prev_size = curr_size

    del expanded
    return best_mask


def _density_region_grow(positions, iords, voxel_size, prev_iords=None,
                         density_threshold=1.5):
    """
    Region-growing clustering with density-based stopping criterion.

    Voxelises particle positions, seeds from the centroid of prev_iords
    (or the densest voxel), then grows outward one shell at a time via
    binary_dilation. Only occupied voxels are added to the region.

    Growth stops when adding the next shell would cause the region's
    average density (particles / occupied voxels) to exceed
    density_threshold × the current region density.

    Parameters
    ----------
    positions         : (N, 3) float array – positions in kpc
    iords             : (N,) int array
    voxel_size        : float – voxel edge in kpc
    prev_iords        : array-like or None – iords from previous snapshot for seeding
    density_threshold : float – max allowed density ratio when adding a shell (default 1.5)

    Returns
    -------
    mask : (N,) bool array, or None if no particles found
    """
    from scipy.ndimage import binary_dilation

    if len(positions) == 0:
        return None

    vx = np.floor(positions[:, 0] / voxel_size).astype(np.int64)
    vy = np.floor(positions[:, 1] / voxel_size).astype(np.int64)
    vz = np.floor(positions[:, 2] / voxel_size).astype(np.int64)

    ox, oy, oz = int(vx.min()), int(vy.min()), int(vz.min())
    gx = (vx - ox).astype(np.int32)
    gy = (vy - oy).astype(np.int32)
    gz = (vz - oz).astype(np.int32)
    del vx, vy, vz

    nx, ny, nz = int(gx.max()) + 1, int(gy.max()) + 1, int(gz.max()) + 1

    # Count grid: number of particles per voxel
    count_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    np.add.at(count_grid, (gx, gy, gz), 1)
    occupied = count_grid > 0

    # Find seed voxel
    if prev_iords is not None and len(prev_iords) > 0:
        in_prev = np.isin(iords, prev_iords)
        if in_prev.any():
            centroid = positions[in_prev].mean(axis=0)
            sx = int(np.floor(centroid[0] / voxel_size)) - ox
            sy = int(np.floor(centroid[1] / voxel_size)) - oy
            sz = int(np.floor(centroid[2] / voxel_size)) - oz
            sx = np.clip(sx, 0, nx - 1)
            sy = np.clip(sy, 0, ny - 1)
            sz = np.clip(sz, 0, nz - 1)
            seed = (sx, sy, sz)
        else:
            seed = np.unravel_index(count_grid.argmax(), count_grid.shape)
    else:
        seed = np.unravel_index(count_grid.argmax(), count_grid.shape)

    # Initialise region with seed voxel
    region = np.zeros((nx, ny, nz), dtype=bool)
    region[seed] = True

    region_particles = int(count_grid[seed])
    region_voxels = 1
    current_density = float(region_particles) / region_voxels if region_voxels > 0 else 0.0

    structure3 = np.ones((3, 3, 3), dtype=bool)

    # Grow shell by shell
    for step in range(max(nx, ny, nz)):
        # Dilate region by one shell
        expanded = binary_dilation(region, structure=structure3, iterations=1)
        new_shell = expanded & ~region & occupied

        if not new_shell.any():
            break

        shell_particles = int(count_grid[new_shell].sum())
        shell_voxels = int(new_shell.sum())

        new_total_p = region_particles + shell_particles
        new_total_v = region_voxels + shell_voxels
        new_density = float(new_total_p) / new_total_v

        if current_density > 0 and new_density > current_density * density_threshold:
            print(f'    region grow: shell {step + 1} would increase density '
                  f'{current_density:.2f} → {new_density:.2f} '
                  f'(×{new_density / current_density:.2f}), stopping')
            break

        region |= new_shell
        region_particles = new_total_p
        region_voxels = new_total_v
        current_density = new_density

    print(f'    region grow: {region_particles} particles in {region_voxels} voxels '
          f'after {step + 1} shells')

    # Map back to particles
    mask = region[gx, gy, gz]
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
    voxel_size_kpc=0.08,
    max_degree=20,
    size_jump=2.0,
    track_cluster_file=None,
    max_instances=None,
    density_threshold=1.5,
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
        voxel_size_kpc    - voxel edge length in kpc (default 0.08)
        max_degree        - maximum connectivity radius in voxel steps (default 20)
        size_jump         - cluster size ratio that signals satellite absorption (default 2.0)

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
    if max_instances is not None:
        instance_files = instance_files[:max_instances]
    n_instances = len(instance_files)
    print(f'Found {n_instances} instance files in {tagged_dir}')

    # AHF setup — track_cluster_file supersedes path_AHF_halonums when provided
    if track_cluster_file is not None:
        import h5py
        _tc_rows = []
        with h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_rows.append({
                        'snapshot':    snap,
                        'AHF halonum': int(f[snap]['main']['halonum'][()]),
                    })
        AHF_halonums = pd.DataFrame(_tc_rows)
        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
        AHF_centers  = None
        print(f'Loaded track_cluster file: {len(AHF_halonums)} snapshots, using AHF halonums')
    else:
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
                    if len(halonum_snap) == 0:
                        raise KeyError(f'Snapshot {outputs[i]} not found in track_cluster file')
                    h = DMOparticles.halos(halo_numbers='v1')[int(halonum_snap[0])]
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

            # Density region growing with this instance's own PrevVoxelIords
            if use_clustering:
                pos_k   = np.array(particle_sel_k['pos'])
                iords_k = np.asarray(particle_sel_k['iord'])
                mask_k  = _density_region_grow(
                    pos_k, iords_k, float(voxel_size_kpc),
                    prev_iords=PrevVoxelIords[k] if len(PrevVoxelIords[k]) > 0 else None,
                    density_threshold=density_threshold,
                )
                if mask_k is None or mask_k.sum() == 0:
                    continue
                particle_sel_k    = particle_sel_k[mask_k]
                PrevVoxelIords[k] = np.asarray(particle_sel_k['iord'])
                # restrict insitu to the cluster
                if len(parts_insitu_k) > 0:
                    parts_insitu_k = parts_insitu_k[
                        np.isin(parts_insitu_k['iord'], particle_sel_k['iord'])
                    ]

            iords_k_arr = np.asarray(particle_sel_k['iord'])
            masses_k = data_grouped_k.loc[iords_k_arr, 'mstar'].values

            # Centre on the cluster (insitu if available, else all tagged)
            if len(parts_insitu_k) > 0:
                insitu_iords_arr = np.asarray(parts_insitu_k['iord'])
                masses_insitu_k = data_insitu_k.loc[insitu_iords_arr, 'mstar'].values
                cen_stars_k = calc_3D_cm(parts_insitu_k, masses_insitu_k)
            else:
                cen_stars_k = calc_3D_cm(particle_sel_k, masses_k)

            particle_sel_k['pos'] -= cen_stars_k

            distances_k = np.sqrt(np.array(particle_sel_k['x'])**2 + np.array(particle_sel_k['y'])**2)
            sort_idx_k  = np.argsort(distances_k)
            sorted_dists_k = distances_k[sort_idx_k]
            sorted_masses_k = masses_k[sort_idx_k]
            cumsum_k = np.cumsum(sorted_masses_k)
            R_half_k = float(sorted_dists_k[np.searchsorted(cumsum_k, cumsum_k[-1] * 0.5)])

            lum_k = produce_lums_grouped(dt_all_k, iords_k_arr, t_val)
            # halflight via cumsum (replaces slow iterative pynbody LowPass binary search)
            lum_sort_k = lum_k[sort_idx_k]
            cumsum_lum_k = np.cumsum(lum_sort_k)
            hlight_k = float(sorted_dists_k[np.searchsorted(cumsum_lum_k, cumsum_lum_k[-1] * 0.5)]) if cumsum_lum_k[-1] > 0 else float('nan')

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


def calculate_reffs_hydro_stars(
    HYDROsim,
    pynbody_path=None,
    halo_number=0,
    output_fname=None,
    save_to_file=True,
    use_clustering=True,
    use_ahf=False,
    voxel_size_kpc=0.08,
    max_degree=20,
    size_jump=2.0,
    track_cluster_file=None,
    density_threshold=1.5,
):
    '''
    Calculate half-light and half-mass radii directly from HYDRO stellar particles.

    No tagging required — deterministic, single output CSV.
    Applies edge_tangos_properties metallicity corrections before luminosity calculation,
    then uses pynbody.analysis.luminosity.half_light_r for the halflight radius.
    Voxel clustering isolates the main galaxy from satellites inside r200c.

    Inputs:
        HYDROsim       - tangos simulation object (HYDRO)
        pynbody_path   - path to snapshot data (defaults to config)
        halo_number    - halo index (default 0)
        output_fname   - output CSV path (default: <sim_name>_hydro_reffs.csv)
        save_to_file   - write CSV incrementally (default True)
        use_clustering - use iterative voxel clustering to exclude satellites (default True)
        use_ahf        - use AHF catalogue instead of HOP (default False)
        voxel_size_kpc - voxel edge length in kpc (default 0.08)
        max_degree     - max voxel connectivity radius in steps (default 20)
        size_jump      - cluster size ratio signalling satellite absorption (default 2.0)

    Returns:
        DataFrame with columns: t, z, reff, halflight, kravtsov
    '''
    try:
        import edge_tangos_properties as etp
    except ImportError:
        etp = None
        print('Warning: edge_tangos_properties not found — metallicity corrections will be skipped')

    if pynbody_path is None:
        pynbody_path = config.get_path('pynbody_path')

    simname = HYDROsim.path
    if output_fname is None:
        output_fname = simname.replace('/', '_') + '_hydro_reffs.csv'

    # Load track_cluster HDF5 if supplied
    _tc_halonum_map = None
    if track_cluster_file is not None:
        import h5py as _h5py
        _tc_data = {}
        with _h5py.File(track_cluster_file, 'r') as f:
            for snap in f.keys():
                if 'main' in f[snap] and 'halonum' in f[snap]['main']:
                    _tc_data[snap] = int(f[snap]['main']['halonum'][()])
        _tc_halonum_map = _tc_data
        print(f'Loaded track_cluster file: {len(_tc_halonum_map)} snapshots, using AHF halonums')

    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(HYDROsim, halo_number + 1)

    split    = simname.split('_')
    halonum  = split[0][4:]
    HYDROname = simname  # full tangos path

    # Resume: build set of already-processed output names
    processed_outputs = set()
    stored_reff      = np.array([])
    stored_reff_z    = np.array([])
    stored_time      = np.array([])
    kravtsov_r       = np.array([])
    lum_halflight    = np.array([])
    t_to_output      = {round(float(t_all[i]), 8): outputs[i] for i in range(len(outputs))}

    if save_to_file and os.path.isfile(output_fname):
        try:
            existing = pd.read_csv(output_fname, index_col=0)
            if len(existing) > 0:
                stored_reff     = existing['reff'].values
                stored_reff_z   = existing['z'].values
                stored_time     = existing['t'].values
                kravtsov_r      = existing['kravtsov'].values
                lum_halflight   = existing['halflight'].values
                for tv in existing['t'].values:
                    out_name = t_to_output.get(round(float(tv), 8))
                    if out_name is not None:
                        processed_outputs.add(out_name)
                print(f'Resuming: {len(existing)} snapshots already done')
        except Exception as e:
            print(f'Could not read existing output ({e}), starting fresh')

    PrevVoxelIords = np.array([])

    # Main loop — z=0 first (backwards)
    for i in range(len(outputs))[::-1]:
        gc.collect()

        if outputs[i] in processed_outputs:
            print(f'Skipping {outputs[i]} (already done)')
            continue

        print('Current snapshot -->', outputs[i])

        t_val = t_all[i]
        z_val = red_all[i]

        hDMO = tangos.get_halo(HYDROname + '/' + outputs[i] + '/halo_' + str(halonums[i]))

        simfn = join(pynbody_path, outputs[i])
        try:
            HYDROparticles = pynbody.load(simfn)
        except Exception as e:
            print(f'--> Failed to load snapshot, skipping! Error: {e}')
            continue

        # Apply etp metallicity corrections before any filtering
        if etp is not None:
            try:
                etp.stars.StellarProperty._ensure_ramses_metal_are_corrected(HYDROparticles)
            except Exception as e:
                print(f'  etp correction failed ({e}), continuing without')

        try:
            if _tc_halonum_map is not None and outputs[i] in _tc_halonum_map:
                pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
                h = HYDROparticles.halos(halo_numbers='v1')[int(_tc_halonum_map[outputs[i]])]
            elif use_ahf:
                pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
                h = HYDROparticles.halos(halo_numbers='v1')[int(halonums[i]) - 1]
            else:
                hop_cat = pynbody.halo.hop.HOPCatalogue(HYDROparticles)
                h = hop_cat[int(halonums[i]) - 1]

            HYDROparticles.physical_units()
            pynbody.analysis.halo.center(h)
            pynbody.analysis.angmom.faceon(h.dm)
        except Exception as e:
            print(f'Centering failed: {e}, skipping')
            continue

        try:
            kravtsov = hDMO['r200c'] * 0.02
        except KeyError:
            kravtsov = float('nan')

        # Use stellar particles from the AHF halo — voxel clustering will isolate the main galaxy
        stars = h.st

        if len(stars) == 0:
            print('  No stellar particles in snapshot, skipping')
            continue

        # Filter out zero-iron-metallicity stars
        if etp is not None:
            try:
                good_mask = etp.stars.AbundanceRatios._mask_stars_with_zero_iron_metallicity(stars)
                stars = stars[good_mask]
            except Exception as e:
                print(f'  zero-iron mask failed ({e}), skipping filter')

        if len(stars) == 0:
            print('  No stellar particles after metallicity filter, skipping')
            continue

        # Density region growing to isolate main galaxy from satellites
        if use_clustering:
            pos_st   = np.array(stars['pos'])
            iords_st = np.asarray(stars['iord'])
            mask_st  = _density_region_grow(
                pos_st, iords_st, float(voxel_size_kpc),
                prev_iords=PrevVoxelIords if len(PrevVoxelIords) > 0 else None,
                density_threshold=density_threshold,
            )
            if mask_st is None or mask_st.sum() == 0:
                print('  Region growing returned empty cluster, skipping')
                continue
            cluster_stars  = stars[mask_st]
            PrevVoxelIords = np.asarray(cluster_stars['iord'])
        else:
            cluster_stars = stars

        # Centre on cluster
        cen = calc_3D_cm(cluster_stars, cluster_stars['mass'])
        cluster_stars['pos'] -= cen

        # Half-light radius via pynbody SSP (etp corrections already applied)
        try:
            hlight = pynbody.analysis.luminosity.half_light_r(cluster_stars, band='V')
        except Exception as e:
            print(f'  half_light_r failed ({e}), storing NaN')
            hlight = float('nan')

        # Half-mass radius (cylindrical, sorted cumsum)
        rxy = np.sqrt(cluster_stars['x']**2 + cluster_stars['y']**2)
        sorted_idx    = np.argsort(rxy)
        sorted_rxy    = np.asarray(rxy)[sorted_idx]
        sorted_masses = np.asarray(cluster_stars['mass'])[sorted_idx]
        cumsum_mass   = np.cumsum(sorted_masses)
        R_half        = float(sorted_rxy[np.where(cumsum_mass >= cumsum_mass[-1] / 2)[0][0]])

        cluster_stars['pos'] += cen

        stored_reff    = np.append(stored_reff,    R_half)
        stored_reff_z  = np.append(stored_reff_z,  z_val)
        stored_time    = np.append(stored_time,    t_val)
        kravtsov_r     = np.append(kravtsov_r,     kravtsov)
        lum_halflight  = np.append(lum_halflight,  hlight)
        processed_outputs.add(outputs[i])

        if save_to_file:
            df_out = pd.DataFrame({
                'halflight': lum_halflight,
                'reff':      stored_reff,
                'z':         stored_reff_z,
                't':         stored_time,
                'kravtsov':  kravtsov_r,
            })
            df_out.to_csv(output_fname)
            print(f'  Wrote {output_fname}')

        del HYDROparticles

    df_final = pd.DataFrame({
        'halflight': lum_halflight,
        'reff':      stored_reff,
        'z':         stored_reff_z,
        't':         stored_time,
        'kravtsov':  kravtsov_r,
    })
    if save_to_file:
        df_final.to_csv(output_fname)

    return df_final


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

    
