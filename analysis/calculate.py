import numpy as np
import pynbody
import pynbody.filt as f
from pynbody.analysis.luminosity import get_current_ssp_table
#from pynbody.analysis.luminosity import SSPTable
from config import config 

def calc_3D_cm(particles,masses):

    x_cm = sum(particles['x']*masses)/sum(masses)

    y_cm = sum(particles['y']*masses)/sum(masses)

    z_cm = sum(particles['z']*masses)/sum(masses)

    return np.asarray([x_cm,y_cm,z_cm])


def projected_halfmass_radius(particles, tagged_masses):
    '''
    inputs: 

    particles - pynbody particle data 

    tagged_masses - stellar masses tagged 

    
    Output: 
    
    R_half - 2D stellar projected half-mass radius 

    '''
    particle_distances =  np.sqrt(particles['x']**2 + particles['y']**2 )
  
    idxs_distances_sorted = np.argsort(particle_distances)

    sorted_distances = particle_distances[idxs_distances_sorted]
                
    sorted_massess = tagged_masses[idxs_distances_sorted]
                
    cumilative_sum = np.cumsum(sorted_massess)

    R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]

    return R_half



def calc_luminosity(particle_ages,masses):
    
    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)

    # Assumes stars have a metallicity of -2 (1/100 L_sun)
    metals = np.ones(len(particle_ages))*(3*10**(-4))
    # Age in yrs
    particle_ages = particle_ages * (10**9)

    ## calculating v-band mags

    #lums_data = np.load(pynbody.analysis.luminosity._default_ssp_file[0])
    #lums = {'ages':np.log10(lums_data[:,0]), 'mets': np.log10(lums_data[:,1]), 'v':lums_data[:,5]}
    
    #lums = np.load('/scratch/dp191/shared/python/anaconda3/envs/py311/lib/python3.11/site-packages/pynbody/analysis/cmdlum.npz')
    lums = get_current_ssp_table()

    age_star = pynbody.array.SimArray(particle_ages, units = 'yr')
    age_star[age_star<1.0] = 1.0

    '''
    age_grid = lums['ages']*(10**(-9))
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(age_grid))] = np.min(age_grid)
    age_star[np.where(age_star > np.max(age_grid))] = np.max(age_grid)

    metals[np.where(metals < np.min(np.log10(lums['mets'])))] = np.min(np.log10(lums['mets']))
    metals[np.where(metals > np.max(np.log10(lums['mets'])))] = np.max(np.log10(lums['mets']))
    
    print(age_grid)
    
    #age_grid = lums['ages']
    met_grid = np.log10(lums['mets'])
    mag_grid = lums['v']

    output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)
    '''
    output_mags = lums.interpolate(np.log10(particle_ages) , np.log10(metals), 'V')
    # calculating luminosities

    vals = output_mags - 2.5 * np.log10(masses)
    
    v_mag_sun = 4.8

    lum_msol = 10.0 ** ((v_mag_sun - vals)/2.5)

    #print('luminosity (Msol): ',list(lum_msol))
    
    return lum_msol

def calc_mags_tagged(particle_ages,masses,band='V'):
    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)


    # Assumes stars have a metallicity of -2 (1/100 L_sun)
    metals = np.ones(len(particle_ages))*(3*10**(-4))
    # Age in yrs
    particle_ages = particle_ages * (10**9)

    ## calculating v-band mags

    #lums_data = np.load(pynbody.analysis.luminosity._default_ssp_file[0])
    #lums = {'ages':np.log10(lums_data[:,0]), 'mets': np.log10(lums_data[:,1]), 'v':lums_data[:,5]}

    #lums = np.load('/scratch/dp191/shared/python/anaconda3/envs/py311/lib/python3.11/site-packages/pynbody/analysis/cmdlum.npz')
    lums = get_current_ssp_table()

    age_star = pynbody.array.SimArray(particle_ages, units = 'yr')
    age_star[age_star<1.0] = 1.0


    output_mags = lums.interpolate(np.log10(particle_ages) , np.log10(metals), 'V')
    # calculating luminosities

    vals = output_mags - 2.5 * np.log10(masses)

    return vals


def calculate_x(pos,ndim):
    return ((pos[:, 0:ndim] ** 2).sum(axis=1)) ** (1, 2)



def calc_sb(tagged_DMO_particles, lums, bin_type='lin',nbins=100,ndims=2):

    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)

    
    # SB = -2.5 Log10(L/ (4 pi d^2)) + 26.4 ?
    if ndims==3:
        particles_rdists = tagged_DMO_particles['r'].in_units('pc')
        
    if ndims==2: 
        particles_rdists = np.sqrt(tagged_DMO_particles["x"].in_units('pc')**2 + tagged_DMO_particles["y"].in_units('pc')**2)
    else:
        print("Invalid value supplied for ndims (must be one of 2 or 3)")

    #particles_rdists = calculate_x(tagged_DMO_particles['pos'].in_units('pc'),ndims)
    #print('rad max:',max(particles_rdists),'min:',min(particles_rdists))

    # generate bin edges
    if bin_type == 'log':
        bin_edges = np.logspace(np.log10(min(particles_rdists)), np.log10(max(particles_rdists)), num=nbins+1)
        
    if bin_type == 'lin':
        bin_edges = np.linspace(min(particles_rdists), max(particles_rdists), num=nbins + 1)

    if bin_type == 'eq':
        bin_edges = pynbody.util.equipartition(particles_rdists, nbins, min(particles_rdists), max(particles_rdists))
        
    
    print(bin_edges,"bin_edges_generated")
    #print('bin max:',max(bin_edges),'min:',min(bin_edges))
    
    # make histogram weighted by luminosity (ref lum = 1)
    #calculated_mags = 4.8 - 2.5*np.log10(lums)

    #part_luminosity = 10.0 ** (-0.4 * calculated_mags)
    
    part_luminosity = lums
    
    hist,bin_edges_hist = np.histogram(particles_rdists,bins=bin_edges,weights=part_luminosity)
    print(bin_edges_hist,'bin_edges_hist')

    #print(hist,bin_edges_hist,bin_edges,'bin edges')

    if ndims == 2:
        # bin_edges are in pc, binsize in pc^2 
        binsize = np.pi * (bin_edges_hist[1:] ** 2 - bin_edges_hist[:-1] ** 2)
    
    else:
        binsize = 4. / 3. * np.pi * (bin_edges_hist[1:] ** 3 - bin_edges_hist[:-1] ** 3)

    
    #sqarcsec_in_bin = binsize #/ 2.3504430539466191e-09

    # Surface Brightness = -2.5 log10(Luminosity/ area) 
    # S = m_apprent + log10(Area)

    # S(mags/arcsecond^2) = M_sun + 21.572 - 2.5Log10(S(L_sun/pc^2))

    #surfb = -2.5 * np.log10(hist / binsize)
    surfb = hist / binsize

    #print(surfb,"<-----surfb")

    # S(mags/arcsecond^2)
    S_observer_units = 4.83 + 21.572 - 2.5*np.log10(surfb)

    #print(surfb,'hist bins ')
    rbins = (np.asarray(bin_edges_hist[1:]) + np.asarray(bin_edges_hist[:-1]))/2
    
    print(rbins,"rbins")
    
    return S_observer_units,rbins,surfb 


    





def calc_lum_hydro(ages_h,masses_h,metals_h):

    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)


    # Assumes stars have a metallicity of -2 (1/100 L_sun)
    metals = metals_h
    ages_h = ages_h
    ## calculating v-band mags

    #lums_data = np.genfromtxt(pynbody.analysis.luminosity._default_ssp_file[0])
    #lums = {'ages':np.log10(lums_data[:,0]), 'mets': np.log10(lums_data[:,1]), 'v': lums_data[:,5]}
    #lums = np.load('/scratch/dp191/shared/python/anaconda3/envs/py311/lib/python3.11/site-packages/pynbody/analysis/cmdlum.npz')
    
    lums = get_current_ssp_table()
    
    age_star = ages_h 
    #pynbody.array.SimArray(ages_h , units = 'yr')
    
    '''
    age_grid = lums['ages']*(10**(-9))
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(age_grid))] = np.min(age_grid)
    age_star[np.where(age_star > np.max(age_grid))] = np.max(age_grid)
    metals[np.where(metals < np.min(np.log10(lums['mets'])))] = np.min(np.log10(lums['mets']))
    metals[np.where(metals > np.max(np.log10(lums['mets'])))] = np.max(np.log10(lums['mets']))

    
    #age_grid = lums['ages']
    met_grid = np.log10(lums['mets'])
    mag_grid = lums['v']
    
    print('age grid',age_grid)
    '''
    
    output_mags = lums.interpolate(np.log10(ages_h) , np.log10(metals), 'V')
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)

    # calculating luminosities

    vals = output_mags - 2.5 * np.log10(masses_h)

    v_mag_sun = 4.8

    lum_msol = 10.0 ** ((v_mag_sun - vals)*0.4)

    #print('luminosity (Msol): ',list(lum_msol))

    return lum_msol

def calc_mags_hydro(ages_h,masses_h,metals_h):

    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)


    # Assumes stars have a metallicity of -2 (1/100 L_sun)
    metals = metals_h
    ages_h = ages_h*(10**9)
    ## calculating v-band mags

    #lums_data = np.genfromtxt(pynbody.analysis.luminosity._default_ssp_file[0])
    #lums = {'ages':np.log10(lums_data[:,0]), 'mets': np.log10(lums_data[:,1]), 'v': lums_data[:,5]}
    #lums = np.load('/scratch/dp191/shared/python/anaconda3/envs/py311/lib/python3.11/site-packages/pynbody/analysis/cmdlum.npz')

    lums = get_current_ssp_table()

    age_star = pynbody.array.SimArray(ages_h , units = 'yr')

    '''
    age_grid = lums['ages']*(10**(-9))
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(age_grid))] = np.min(age_grid)
    age_star[np.where(age_star > np.max(age_grid))] = np.max(age_grid)
    metals[np.where(metals < np.min(np.log10(lums['mets'])))] = np.min(np.log10(lums['mets']))
    metals[np.where(metals > np.max(np.log10(lums['mets'])))] = np.max(np.log10(lums['mets']))


    #age_grid = lums['ages']
    met_grid = np.log10(lums['mets'])
    mag_grid = lums['v']

    print('age grid',age_grid)
    '''

    output_mags = lums.interpolate(np.log10(ages_h) , np.log10(metals), 'V')
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)

    # calculating luminosities

    vals = output_mags - 2.5 * np.log10(masses_h)


    return vals 




def calc_ages(tagged_particle_df, t_current):

    tform = tagged_particle_df.groupby(['iords']).first()['t'].values

    age = t_current-tform

    return age 




def produce_lums_grouped(df,present_iords,t_snap):

    #tagging_info_for_each_particle = df.groupby(['iords'])

    ages = t_snap - df['t'].values 
    masses_st =df['mstar'].values

    lum_vals = calc_luminosity(ages,masses_st)

    df['lums'] = lum_vals

    lums_df = df.groupby(['iords']).sum()['lums']

    lums_for_part = np.asarray([lums_df.loc[iord] for iord in present_iords])
    '''
    for particle_id_tag in present_iords:

        group = tagging_info_for_each_particle.get_group(particle_id_tag)

        group.sort_values(by="t",ascending=True,inplace=True)
        
        masses_particle = group['mstar'] - group['mstar'].shift(1)

        masses_particle.iloc[0] = group['mstar'].iloc[0] 

        #mass_form = np.append(mass_form,masses_particle)
        tgrp = t_snap - group['t'].values
        #ages_from_tag = np.append(ages_from_tag,tgrp)
        lum_grp = np.sum(np.asarray(calc_luminosity(tgrp,masses_particle)))
        lums_for_part = np.append(lums_for_part,lum_grp)
    '''

    
    return lums_for_part 


def calc_tot_lum(particle_ages,masses):
    

    lums = calc_luminosity(particle_ages,masses)

    halo_luminosity = np.sum(lums)
    
    return halo_luminosity 


def calc_halflight(sim,lum_for_each_iord,band='v',cylindrical=False):
    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)


    '''
    Assumes ordering of ages_st is the same as sim_particles
    '''
    
    half_l = np.sum(lum_for_each_iord) * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'

    max_high_r = np.max(sim.dm[coord])
    test_r = 0.5 * max_high_r
    
    testrf = f.LowPass(coord, test_r)
    min_low_r = 0.0

    #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
    #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
    
    test_l = np.sum(lum_for_each_iord[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])])
    
    it = 0

    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 20):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5

        testrf = f.LowPass(coord, test_r)
        #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        test_l = np.sum(lum_for_each_iord[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])])
            
        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r
    
    return test_r




def calc_halflight_hydro(sim,lum_for_each_iord,band='v',cylindrical=False):

    #### function adapted from pynbody pynbody.analysis.luminosity 
    #### (see https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html)

    
    '''
    Assumes ordering of ages_st is the same as sim_particles
    '''

    half_l = np.sum(lum_for_each_iord) * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'

    max_high_r = np.max(sim.st[coord])
    test_r = 0.5 * max_high_r

    testrf = f.LowPass(coord, test_r)
    min_low_r = 0.0

    #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
    #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]

    test_l = np.sum(lum_for_each_iord[np.isin(sim.st['iord'],sim.st[testrf]['iord'])])

    it = 0

    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 20):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5

        testrf = f.LowPass(coord, test_r)
        #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        test_l = np.sum(lum_for_each_iord[np.isin(sim.st['iord'],sim.st[testrf]['iord'])])

        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r

    return test_r



def bootstrap_stat(data,stat_func,num_resamples=100):
    
    '''
    Calculates a bootstrapped statistic 
    
    num_resamples = number of times we want to draw a sample from the data 
    
    stat_func = a function that will be applied to the data during each resampling run 

    data = must be a numpy array of non-zero size 

    '''
    stat_values = np.array([])
    
    assert data.shape[0] > 0, 'Data has size 0'
    
    assert num_resamples >0, 'num_resamples must be > 0'


    for i in range(num_resamples): 
        
        sample = np.random.choice(data,size=data.shape[0])
        
        stat_calculated = stat_func(data)

        stat_values = np.append(stat_values,stat_calculated)


        
    return stat_values 




