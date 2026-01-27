
####### Under Construction #######

import pynbody
import pandas as pd
import numpy as np
import darklight 
import matplotlib.pyplot as plt
import sys
import tangos
import os
import matplotlib.style
import matplotlib as mpl
import seaborn as sns
from darktag.tagging.utils import *
from darktag.analysis.calculate import *
import edge_tangos_properties as etp
from ...config import config 

#mpl.rcParams.update({'text.usetex': False})

pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]


def edge_plot_tagged_vs_hydro_mass_dist(name_of_DMO_simulation, name_of_HYDRO_simulation, file_with_tagged_particles, time_to_plot, plot_type='2D Mass Distribution',label=None):
    
    # Get paths from config
    pynbody_path = config.get_path("pynbody_path")
    tangos_path = config.get_path("tangos_path")    


    # finding mass distribution of tagged particles in DMO simulation 

    split = name_of_DMO_simulation.split('_')

    halo_shortname = split[0]

    tangos.core.init_db(tangos_path+halo_shortname+'.db') 

    DMOsim = tangos.get_simulation(name_of_DMO_simulation)
    
    t_all,red_all,main_halo, halonums, outputs = load_indexing_data(DMOsim,1)

    times_tangos = main_halo.calculate_for_progenitors('t()')[0][::-1]

    time_to_plot = times_tangos[np.argmin(abs(times_tangos-time_to_plot))]
    
    output_number = np.where(times_tangos <= time_to_plot)[0][-1]
  
    d = pd.read_csv(file_with_tagged_particles)

    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).sum()

    tagged_iords = dt.index.values
    tagged_m = dt['mstar'].values
    
    print(os.path.join(pynbody_path,name_of_DMO_simulation,outputs[output_number]))
    s = pynbody.load(os.path.join(pynbody_path,name_of_DMO_simulation,outputs[output_number]))
    
    h = s.halos()[int(halonums[output_number] - 1)]
    
    s.physical_units()
    pynbody.analysis.halo.center(h.dm)
    #pynbody.analysis.angmom.faceon(h.dm)
    r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')
    
    print('DMO R200:',r200_DMO)
    h_r200 = h.dm[h.dm['r'] < r200_DMO]

    selected_parts = h_r200[np.isin(h_r200['iord'],tagged_iords)]

    #idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

    selected_masses = dt.loc[selected_parts['iord']]['mstar']
    

    selected_parts['pos'] -= calc_3D_cm(selected_parts,selected_masses)


    st_ages = time_to_plot - (d[ d['t'] <= time_to_plot ]['t'].values)
    

    #grouped_first = d[ d['t'] <= time_to_plot ].groupby(['iords']).first()
    
    ages_df = pd.DataFrame({'ages':st_ages , 'iords':d[ d['t']<=time_to_plot]['iords'].values, 'mstar': d[ d['t']<=time_to_plot]['mstar'].values})

    #ordered_ages = np.asarray([ ages_df.loc[part_id]['ages'] for part_id in selected_parts['iord'] ])
    
    #ordered_ages = np.asarray([ ages_df.loc[selected_parts['iord'].values]['ages'] ])
    
    lums = calc_luminosity(ages_df['ages'].values,ages_df['mstar'].values)
    
    #Vmags = calc_mags_tagged(ages_df['ages'].values,ages_df['mstar'].values) 
    #print(lums)
    
    ages_df['lums'] = lums
    #ages_df['Vmags'] = Vmags

    lums_particles = ages_df.groupby(['iords']).sum().loc[selected_parts['iord']]['lums'].values
    #Vmags_particles = ages_df.groupby(['iords']).sum().loc[selected_parts['iord']]['Vmags'].values

    data_all_tagged = pd.DataFrame({ 'x':selected_parts['x'], 'y':selected_parts['y'], 'masses':np.asarray(selected_masses), 'lums':lums_particles}) #'Vmags':Vmags_particles})
    
    sb_obs_units,r_sb_bins,sb = calc_sb(selected_parts, data_all_tagged['lums'].values, bin_type='log',nbins=50,ndims=2)

    print(data_all_tagged)

    dataframe_for_hist = pd.DataFrame({ 'x':selected_parts['x'], 'y':selected_parts['y'],'z':selected_parts['z'],'r':selected_parts['r'], 'masses': np.asarray(selected_masses),'lums':lums_particles })
    #, 'ages': np.asarray(ordered_ages)})
    
    dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

    dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)
    dataframe_for_hist['lum_enclosed'] = np.cumsum(dataframe_for_hist['lums'].values)

    # plotting hydro particles 

    sim = tangos.get_simulation(name_of_HYDRO_simulation)

    t_all_H,red_all_h, HYDRO_main_halo, HYDRO_halonums, outputs_HYDRO = load_indexing_data(sim,1)
    
    times_tangos_HYDRO = HYDRO_main_halo.calculate_for_progenitors('t()')[0][::-1]

    halonums_hydro = HYDRO_main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
    
    output_number_HYDRO = np.where(times_tangos_HYDRO <= time_to_plot)[0][-1]
 
    s_hydro = pynbody.load(os.path.join(pynbody_path,name_of_HYDRO_simulation,outputs_HYDRO[output_number_HYDRO]))
    
    h_hydro = s_hydro.halos()[int(halonums_hydro[output_number_HYDRO] - 1)]    

    s_hydro.physical_units() 
    
    etp.stars.StellarProperty._ensure_ramses_metal_are_corrected(s_hydro)
    
    print('Hydro halo ',h_hydro)

    pynbody.analysis.halo.center(h_hydro)
    

    r200_hydro = pynbody.analysis.halo.virial_radius(h_hydro, overden=200, r_max=None, rho_def='critical')
    
    print('R200 Hydro:',r200_hydro)

    stars = s_hydro.st[s_hydro.st['r']<=r200_hydro]
    
    stars = stars[etp.stars.AbundanceRatios._mask_stars_with_zero_iron_metallicity(stars)]

    stars['pos'] -= calc_3D_cm(stars,stars['mass'])

    #mass_ratios = abundances._get_mass_fractions(stars)

    data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})
    
    lum_st = calc_lum_hydro(stars['age'].in_units('yr'),stars['mass'],stars['metals'])
    #mags_st = calc_mags_hydro(stars['age'].in_units('yr'),stars['mass'],stars['metals'])

    data_all_stars['lums'] = lum_st
    #data_all_stars['Vmags'] = mags_st
    
    sb_hydro_obs_units,r_sb_bins_hydro,sb_hydro = calc_sb(stars, data_all_stars['lums'].values, bin_type='log',nbins=50,ndims=2)
    
    lum_hist = pd.DataFrame({'x':stars['x'],'y':stars['y'],'z':stars['z'],'r':stars['r'],'mass':stars['mass'],'lums':lum_st})

    lum_hist = lum_hist.sort_values(by=['r'])

    lum_hist['lum_enc'] = np.cumsum(lum_hist['lums'].values)

    lum_hist['mass_enc'] = np.cumsum(lum_hist['mass'].values)
    print(data_all_stars.head())
    


    if plot_type == "return dataframe of luminosities":
        
        return dataframe_for_hist,lum_hist



    if plot_type == '2D Mass Distribution':

      plt.gca().set_box_aspect(1)
      
      sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,levels=5,cmap="viridis")#,cbar=True) #label='Tagged Stellar Mass')
      sns.kdeplot(data = data_all_stars, x ='x',y='y', weights='masses',fill=False,levels=5,color='white')#,cbar=True)
      '''

      hist, xedges, yedges = np.histogram2d(data_all_tagged['x'], data_all_tagged['y'],bins=100,weights=data_all_tagged['masses'])

      bin_area = abs((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))

      #mass_density = hist/bin_area

      mass_density = hist.T/bin_area

      hist_h, xedges_h, yedges_h = np.histogram2d(data_all_stars['x'], data_all_stars['y'], bins=[xedges, yedges],weights=data_all_stars['masses'])

      mass_density_h = hist_h.T/bin_area
      
      plt.contourf(xedges[:-1], yedges[:-1], mass_density, levels=4)
      
      plt.contour(xedges_h[:-1], yedges_h[:-1], hist_h.T/bin_area,cmap='autumn',levels=4)
      '''
      plt.xlim(-4,4)
      plt.ylim(-4,4)
      
      #plt.colorbar()
      
      plt.title(str(name_of_HYDRO_simulation))


    if plot_type == '2D Pos Distribution':

      plt.gca().set_box_aspect(1)

      #sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,levels=5,cmap="viridis")#,cbar=True) #label='Tagged Stellar Mass')
      #sns.kdeplot(data = data_all_stars, x ='x',y='y', weights='masses',fill=False,levels=5,color='white')#,cbar=True)
      plt.plot(data_all_tagged['x'],data_all_tagged['y'],".",label="tagged")
      plt.plot(data_all_stars["x"],data_all_stars["y"],".",label="HYDRO")
      
      '''

      hist, xedges, yedges = np.histogram2d(data_all_tagged['x'], data_all_tagged['y'],bins=100,weights=data_all_tagged['masses'])

      bin_area = abs((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))

      #mass_density = hist/bin_area

      mass_density = hist.T/bin_area

      hist_h, xedges_h, yedges_h = np.histogram2d(data_all_stars['x'], data_all_stars['y'], bins=[xedges, yedges],weights=data_all_stars['masses'])

      mass_density_h = hist_h.T/bin_area

      plt.contourf(xedges[:-1], yedges[:-1], mass_density, levels=4)

      plt.contour(xedges_h[:-1], yedges_h[:-1], hist_h.T/bin_area,cmap='autumn',levels=4)
      '''
      plt.xlim(-4,4)
      plt.ylim(-4,4)

      #plt.colorbar()

      plt.title(str(name_of_HYDRO_simulation))


  
    if plot_type == '1D Mass Enclosed':
        
        plt.plot(lum_hist['r'].values,lum_hist['mass_enc'].values,c="k")
        #plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values, label=label)
        plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values, label=label)
        #plt.xlim(0,1)
        #plt.xlim(0.1,None)
        #plt.ylim(300,None)
      
        #plt.yscale('log')
        #plt.xscale('log')

        plt.title(str(name_of_HYDRO_simulation))
      
        plt.ylabel("$M_{star}(<r)$ in $M_{\odot}$")
        plt.xlabel('Radial Distance in kpc')

    if plot_type == '1D Luminosity Distribution':

      plt.plot(lum_hist['r'].values,lum_hist['lum_enc']/max(lum_hist['lum_enc'].values),label='HYDRO')
      plt.plot(dataframe_for_hist['r'],dataframe_for_hist['lum_enclosed']/max(dataframe_for_hist['lum_enclosed'].values),label='Tagged')
      
      print('lums enclosed----->',dataframe_for_hist['lum_enclosed'])
      #plt.xlim(0,1)                                                                                                                                                                                       
      plt.legend(frameon=False)
      #plt.ylim(300,None)

      plt.title(str(name_of_HYDRO_simulation))

      plt.ylabel(" $L(<r)$/$L(R_{200})$ ")

      plt.xlabel('Radial Distance in Kpc')
    

    if plot_type == '2D SB profile':
        
        #print(r_sb_bins_hydro,'sb radii')
        #print(r_sb_bins)
        
        plt.plot(r_sb_bins*(10**(-3)),sb,label='Tagged')
        plt.plot(r_sb_bins_hydro*(10**(-3)),sb_hydro,label='HYDRO')
        
        plt.xlabel('Radial Distance kpc')
        plt.ylabel('Surface Brightness $L_{\odot}/pc^{2}$')
    
    if plot_type == '2D SB profile obs units':

        #print(r_sb_bins_hydro,'sb radii')
        #print(r_sb_bins)
        #At 10 pc (distance for absolute magnitudes), 1 arcsec is 10 AU=1/2.06e4 pc (from pynbody docs)
        # so one pc = 2.06e4 arcsec
        arcsec_hydro = r_sb_bins_hydro * 2.06e4 
        arcsec_dmo = r_sb_bins * 2.06e4        
        plt.plot(r_sb_bins*(10**(-3)),sb_obs_units,label=label)
        plt.plot(r_sb_bins_hydro*(10**(-3)),sb_hydro_obs_units,c="k",label="HYDRO")

        plt.xlabel('Radius (kpc)')
        plt.ylabel('Surface Brightness $mag/arcsec^{2}$')        

    if plot_type == '1D Mass Distribution':
      
      plt.hist(dataframe_for_hist['r'].values,weights=dataframe_for_hist['masses'].values,bins=15)
      
      #plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation))
      
      plt.ylabel('Stellar np. in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')

    if plot_type == '1D Density Distribution':

      counts, bin_edges = np.histogram(dataframe_for_hist['r'].values,weights=dataframe_for_hist['masses'].values,bins=20)
      
      bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
      volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
      density = counts / volumes
      
      counts_h, bin_edges_h = np.histogram(stars['r'],weights=stars['mass'],bins=20)

      bin_centers_h = 0.5 * (bin_edges_h[1:] + bin_edges_h[:-1])
      volumes_h = (4/3) * np.pi * (bin_edges_h[1:]**3 - bin_edges_h[:-1]**3)
      density_h = counts_h / volumes_h

      #plt.xlim(0,1)
      
      plt.plot(bin_centers, density, marker='o', linestyle='-',label='Tagged')
      plt.plot(bin_centers_h, density_h, marker='o', linestyle='-',label='Hydro.')
      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation))

      plt.ylabel('Density ($M_{\odot} kpc^{-3}$)')
      plt.xlabel('Radial Distance in Kpc')
      plt.legend()
    if plot_type == '1D Ages Hist':
        plt.hist(dataframe_for_hist['ages'].values,histtype='step',weights=dataframe_for_hist['masses'].values)

        #plt.xlim(0,1)
        
        plt.title(str(name_of_HYDRO_simulation))
        
        plt.title('Stellar Age in Gyr')
                                    

    if plot_type == 'Median Age Vs Radius':

        bins = np.arange(0,1,1/10)
        
        dataframe_for_hist['r_bins'] = pd.cut(dataframe_for_hist['r'],bins)

        df_grp = dataframe_for_hist.groupby(['r_bins']).median()

        df_r_vals = [i.left for i in df_grp.index.values]
        
        plt.bar(df_r_vals,df_grp['ages'])

        plt.xlabel('Radial Dist. in kpc')

        plt.ylabel('Median stellar age Gyr')
        
    
    if plot_type == '2D Luminosity Distribution':
        
        plt.gca().set_box_aspect(1)
        
        levels = np.linspace(0,1,num=6)
        
        print(levels[2:])

        data_all_tagged['lums'] = data_all_tagged['lums']/data_all_tagged['lums'].max()
        
        data_all_stars['lums'] = data_all_stars['lums']/data_all_stars['lums'].max()
        
        print('Hydro_tot:',data_all_stars['lums'].sum())
        print('Tagged_tot:', data_all_tagged['lums'].sum())
        

        sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='lums',fill=True,cmap="inferno",levels=levels,cbar=True,common_norm=True)
        sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='lums',fill=False,color='white',levels=levels,cbar=True,common_norm=True)
        
        '''
        hist, xedges, yedges = np.histogram2d(data_all_tagged['x'], data_all_tagged['y'],bins=50,weights=data_all_tagged['lums'])

        bin_area = (abs(xedges[1]) - abs(xedges[0])) * (abs(yedges[1]) - abs(yedges[0]))

        #mass_density = hist/bin_area

        mass_density = hist.T/bin_area

        hist_h, xedges_h, yedges_h = np.histogram2d(data_all_stars['x'], data_all_stars['y'], bins=[xedges, yedges],weights=data_all_stars['lums'])
        
        bin_area_h = (abs(xedges_h[1]) - abs(xedges_h[0]))*(abs(yedges_h[1]) - abs(yedges_h[0]))

        mass_density_h = hist_h.T/bin_area_h
        
        print('minimum value:',mass_density_h.min())
        #plt.imshow(hist, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        #plt.hexbin(data_all_tagged['x'], data_all_tagged['y'],C=data_all_tagged['lums'].values, gridsize=200,cmap='inferno')
        #sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='lums',fill=True,levels=5,cmap="inferno")#,cbar=True) #label='Tagged Stellar Mass')
        #sns.kdeplot(data = data_all_stars, x ='x',y='y', weights='lums',fill=False,levels=5,color='white')#,cbar=True)

        #plt.hexbin(data_all_stars['x'], data_all_stars['y'],C=data_all_stars['lums'].values, gridsize=200,cmap='inferno')
        
        plt.contourf(xedges[:-1], yedges[:-1], mass_density, vmin = 3*1112, levels=6,cmap='inferno')

        plt.contour(xedges_h[:-1], yedges_h[:-1], mass_density_h,vmin = 3*1112,cmap='Purples',levels=6)
        '''
        
        #plt.colorbar()

        plt.xlim(4,-4)
        plt.ylim(4,-4)
        
        plt.title(str(name_of_HYDRO_simulation))
                                             
    if plot_type != "return dataframe of luminosities":
        
        return 


def plot_tagged_vs_hydro_mass_dist(DMO_halo_particles, HYDRO_halo_particles, file_with_tagged_particles, time_to_plot, plot_type='2D Mass Distribution'):

    d = pd.read_csv(file_with_tagged_particles)

    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()

    tagged_iords = dt.index.values
    tagged_m = dt['mstar'].values
    
    h = DMO_halo_particles
    
    pynbody.analysis.halo.center(h)

    r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')

    selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]


    idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

    selected_masses = [tagged_m[i] for i in idxs_m]

    data_all_tagged = pd.DataFrame({'x':selected_parts['x'],'y':selected_parts['y'], 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = pd.DataFrame({'r':np.sqrt(selected_parts['x']**2+selected_parts['y']**2+selected_parts['z']**2), 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

    dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)

    
    # plotting hydro particles 
    
    h_hydro = HYDRO_halo_particles
    
    pynbody.analysis.halo.center(h_hydro)
    
    stars = h_hydro.st
    
    data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})
        
    
    if plot_type == '2D Mass Distribution':

      plt.gca().set_box_aspect(1)
      
      sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,cmap="viridis",levels=5)
      sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='masses',fill=False,color='white',levels=5)
       
      plt.xlim(4,-4)
      plt.ylim(4,-4)

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
  
    if plot_type == '1D Mass Enclosed':
        
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
      plt.xlim(0,1)
      
      plt.yscale('log')
      plt.xscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel(' $M_{star}(<r)$ in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')

    if plot_type == '1D Mass Distribution':
      
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
    
      plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel('Stellar Mass in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')
    
    return 




def edge_plot_tagged_vs_hydro_angmom_dist():
    print('not yet implemented')
    return 



def plot_tagged_vs_hydro_angmom_dist(DMO_halo_particles,HYDRO_halo_particles,file_with_tagged_particles,time_to_plot,cmd="return plot"):
    
    h = HYDRO_halo_particles
    s = h.st 
    # centering on most massive halo                                                                                                                                                                                                                               
    
    pynbody.analysis.halo.center(h)
    r200_HYDRO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')
    
    stars = s.st[ np.sqrt(s.st['x']**2+s.st['y']**2+s.st['z']**2) <= r200_HYDRO ] 
                                                                                                                                                                                                                  
    rdists = np.sqrt(stars['x']**2+stars['y']**2+stars['z']**2)
    
    jstars = np.sqrt(stars['j'][:,0]**2+stars['j'][:,1]**2+stars['j'][:,2]**2)
    
    #create dataframe                                                                                                                                                                                                                                                  
    df = pd.DataFrame({'r':rdists,'j':jstars, 'mass':stars['mass']})
    
    s_tagged = DMO_halo_particles

    # Stellar Mass weighted median at each radial distance  for tagged particles                                                                                                                                                                                   
    
    d = pd.read_csv(file_with_tagged_particles)
    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()
    
    tagged_iords = dt.index.values
    
    tagged_particles = s_tagged[np.where(np.isin(s_tagged['iord'],tagged_iords)==True)]

    #print(tagged_stars[0]['iord'], np.where(tagged_masses.index.values == tagged_stars[0]['iord']) )                                                                                                                                                              
    
    #print(tagged_masses.loc[tagged_stars[0]['iord']])                                                                                                                                                                                                             
    
    jtagged = np.sqrt(tagged_particles['j'][:,0]**2+tagged_particles['j'][:,1]**2+tagged_particles['j'][:,2]**2)
    
    rtagged = np.sqrt(tagged_particles['x']**2+tagged_particles['y']**2+tagged_particles['z']**2)
    
    #print(tagged_stars['j'].units)                                                                                                                                                                                                                                
    mstar_tagged = [dt.loc[i]['mstar'] for i in tagged_particles['iord']]
    
    dftagged = pd.DataFrame({'r':rtagged,'j':jtagged, 'mass':mstar_tagged })
                                                                                                                                            
    if cmd != "return plot": 

        return dftagged,df 

    else: 
             
        plt.hist(np.asarray(jtagged),weights= np.asarray(mstar_tagged),histtype='step',label="angmom tagging")                                                                                                                                                        
        plt.hist(np.asarray(jstars),weights= np.asarray(stars['mass']),histtype='step',label="Hydro Sim")                                                                                                                                                             
        plt.yscale('log')
    
        return 





