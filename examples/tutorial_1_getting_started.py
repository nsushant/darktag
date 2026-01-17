# import statements 

import pynbody
import tangos

import darktrace as dtrace 
import matplotlib.pyplot as plt 
from config import config 

# specify preference of halo catalog

# if you're working with a single halo catalogue this step can be omitted
# the HOP catalogue used here is 0 indexed

pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
# Loading-in the tangos data 
## tangos database initialization - assumes that the .db file is in the current working directory 
tangos.core.init_db('Halo1459.db')

## loading in the simulation database  
DMO_database = tangos.get_simulation('Halo1459_DMO')

# angular momentum based particle tagging 
df_tagged_particles = dtrace.tag_particles(DMO_database , path_to_particle_data = config.get_path("pynbody_path"), tagging_method = 'angular momentum', free_param_val = 0.001)

# calculate half-mass radii of the tagged stellar populations
df_half_mass_tagged = dtrace.calculate_rhalf(DMO_database, df_tagged_particles, pynbody_path  = config.get_path("pynbody_path"))

# Plotting half-mass radii from tagged populations Vs R_effective from Hydro sims. 
# HYDRO simulation database 
HYDRO_database = tangos.get_simulation('Halo1459_fiducial')

halflight_hydro,time_array_hydro = HYDRO_database.timesteps[-1].halos[0].calculate_for_progenitors('stellar_projected_halflight', 't()')

plt.plot(df_half_mass_tagged['t'],df_half_mass_tagged['reff'],label='Half-mass (tagged)')
plt.plot(time_array_hydro , halflight_hydro , label='Half-light (HYDRO)')
plt.xlabel('Time in Gyr')
plt.ylabel('Radii in kpc')





  
    
