
### Darktrace Package

<img width="1183" alt="image" src="https://github.com/nsushant/particle_tagging_package/assets/64201587/9cd0684d-7a8f-4015-b329-4456a1f3c27b">

Figure 1.  White circle shows the calculated halflight radius. Blue circle shows the virial radius of the dark matter halo and contours show the stellar mass distribution created by particle tagging. 


This is a darktrace package that assigns stellar mass to particles in dark matter only simulations. The primary aim of the project is to accurately reproduce the sizes and stellar mass distributions of dwarf galaxies. 
Two new tagging methods have been developed and tested angainst the generally used binding energy approach. 

1.  Angular momentum based tagging - associates stellar mass with dark matter particles using their angular momenta. 
2.  Spatial tagging - distributes stellar mass across the galaxy based on a Plummer Profile. (particle dark matter mass = stellar mass tagged)
3.  Binding Energy tagging - associates stellar mass with dark matter particles using their binding energies. 

The package has the following dependencies in its current form. 

1. Darklight
2. Pynbody
3. Tangos
4. Numpy
5. Pandas
6. OS
7. csv
8. random
9. Seaborn (only for plotting scripts)
10. Matplotlib (only for plotting scripts)


An example script is available in the 'examples' folder and demonstrates how the modules are used to tag particles in a simulation from the EDGE suite. This is a version of the angular momentum tagging script that has been 
reduced to the barebones. In addition to the above packages, the tagging methods require tangos databases (including merger trees) , pynbody particle data and currently use either AHF or HOP halo catalogues. 



