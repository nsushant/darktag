# Particle Tagging Configuration Tutorial

This tutorial explains how to use the updated particle tagging functions that now use a centralized configuration system.

## Overview

The particle tagging codebase has been updated to use a centralized configuration system instead of hardcoded paths. This makes the code more portable and easier to maintain across different machines and environments.

## Configuration System

### config/config.json
All paths are now centralized in `config/config.json`:

```json
{
    "paths": {
        "tangos_path": "/scratch/dp324/shared/dp101/EDGE/tangos/",
        "pynbody_path": "/scratch/dp324/shared/dp101/EDGE/",
        "manual_halonum_path": "",
        "manual_mstar_path": ""
    },
    "tagging": {
        "method": "angular_momentum",
        "ftag": 0.01 
    },
    "darklight": {
        "n": 500, 
        "DMO_OR_HYDRO": "DMO",
        "poccupied": "all"
    } 
}
```

### config/config.py
The configuration class provides methods to access paths:
- `config.get_path("tangos_path")` - Get tangos database path
- `config.get_path("pynbody_path")` - Get particle data path
- `config.get("tagging", "ftag")` - Get tagging parameters
- `config.get("darklight", "n")` - Get darklight parameters

## Updated Functions

### 1. Angular Momentum Tagging Functions

#### `angmom_tag_over_full_sim(DMOsim, halonumber=1, free_param_value=0.01, pynbody_path=None, ...)`

**How it works:**
- Loads a tangos simulation and performs angular momentum-based particle tagging
- If `pynbody_path` is None, automatically uses `config.get_path("pynbody_path")`
- Tags particles based on lowest angular momentum within each halo
- Handles both insitu and accreted star tagging

**Example usage:**
```python
from darktrace.tagging.angular_momentum_tagging import angmom_tag_over_full_sim
import tangos

# Load simulation
tangos.init_db(join(config.get_path("tangos_path"), "Halo1459.db"))
DMOsim = tangos.get_simulation("Halo1459_DMO")

# Run tagging (uses config path automatically)
df_tagged = angmom_tag_over_full_sim(DMOsim, halonumber=1, free_param_value=0.01)
```

#### `angmom_tag_over_full_sim_recursive(DMOsim, tstep, halonumber, ...)`

**How it works:**
- Recursively tags accreting halos down the merger tree
- Each accreted halo gets tagged over its entire lifetime
- Prevents double-tagging by tracking which halos have been processed
- Uses config paths automatically when not provided

**Key features:**
- Prevents infinite recursion by tracking tagged halo paths
- Handles complex merger trees
- Supports both DMO and HYDRO simulations

### 2. Binding Energy Tagging Functions

#### `BE_tag_over_full_sim(DMOsim, halonumber, free_param_value=0.01, PE_file=None, pynbody_path=None, ...)`

**How it works:**
- Tags particles based on binding energy (most bound particles first)
- Calculates potentials and kinetic energies for particles
- Uses config paths when pynbody_path is None
- Supports potential energy from file or direct calculation

**Example usage:**
```python
from darktrace.tagging.binding_energy_tagging import BE_tag_over_full_sim

# Run binding energy tagging
df_tagged = BE_tag_over_full_sim(DMOsim, halonumber=1, free_param_value=0.01)
```

### 3. Spatial Tagging Functions

#### `spatial_tag_over_full_sim(DMOsim, pynbody_path=None, occupation_frac='all', ...)`

**How it works:**
- Tags particles based on spatial position (e.g., within specific radius)
- Uses config paths automatically
- Supports various occupation fractions for dark matter halos

### 4. Wrapper Functions

#### `tag_particles(DMO_database, path_to_particle_data=None, tagging_method='angular momentum', ...)`

**How it works:**
- Unified interface to all tagging methods
- Automatic path resolution using config system
- Supports multiple tagging methods:
  - `'angular momentum'` - Standard angular momentum tagging
  - `'angular momentum recursive'` - Recursive tagging with merger handling
  - `'spatial'` - Position-based tagging

**Example usage:**
```python
from darktrace.tagging.tagging_wrapper_func import tag_particles

# Uses config path automatically
df_tagged = tag_particles(DMO_database, tagging_method='angular momentum', free_param_val=0.01)
```

#### `calculate_reffs_over_full_sim(DMOsim, particles_tagged, pynbody_path=None, ...)`

**How it works:**
- Calculates effective radii of tagged stellar populations
- Uses config paths automatically
- Supports both AHF and HOP halo catalogues
- Calculates half-mass radii and half-light radii

## Configuration for Different Environments

### Local Development
Update `config/config.json` with your local paths:
```json
{
    "paths": {
        "tangos_path": "/path/to/your/tangos/databases/",
        "pynbody_path": "/path/to/your/particle/data/"
    }
}
```

### HPC/Cluster Systems
Set paths appropriate for your system:
```json
{
    "paths": {
        "tangos_path": "/scratch/username/shared/simulations/tangos/",
        "pynbody_path": "/scratch/username/shared/simulations/"
    }
}
```

## Best Practices

### 1. Always Use Config System
```python
# Good - uses config
pynbody_path = config.get_path("pynbody_path")

# Avoid - hardcoded paths
pynbody_path = "/some/hardcoded/path"
```

### 2. Parameter Defaults
Functions have sensible defaults and use config when paths are None:
```python
# This will use config automatically
df_tagged = angmom_tag_over_full_sim(DMOsim)

# This overrides config path
df_tagged = angmom_tag_over_full_sim(DMOsim, pynbody_path="/custom/path")
```

### 3. Error Handling
The updated functions include proper error handling:
- Missing simulation files are skipped with warnings
- Invalid halo numbers are handled gracefully
- Path resolution issues provide clear error messages

## Migration Guide

### From Old (Hardcoded Paths)
```python
# Old way
tangos.core.init_db("/scratch/dp101/shared/EDGE/tangos/Halo1459.db")
simfn = join("/scratch/dp101/shared/EDGE/", "output_00000")
```

### To New (Config System)
```python
# New way
tangos.core.init_db(join(config.get_path("tangos_path"), "Halo1459.db"))
simfn = join(config.get_path("pynbody_path"), "output_00000")
```

## Troubleshooting

### Common Issues

1. **Path not found**: Check `config/config.json` paths are correct
2. **Database errors**: Ensure tangos_path points to valid .db files
3. **Particle data missing**: Verify pynbody_path contains simulation outputs
4. **Permission errors**: Make sure paths are accessible with current permissions

### Debug Mode
Add debug prints to verify path resolution:
```python
print(f"Using tangos path: {config.get_path('tangos_path')}")
print(f"Using pynbody path: {config.get_path('pynbody_path')}")
```

## Advanced Usage

### Custom Configuration
You can extend the config system for additional parameters:
```json
{
    "paths": {
        "tangos_path": "/path/to/tangos/",
        "pynbody_path": "/path/to/particle/data/",
        "custom_output_path": "/path/to/outputs/"
    },
    "custom_section": {
        "parameter": "value"
    }
}
```

Access custom parameters:
```python
custom_path = config.get_path("custom_output_path")
custom_param = config.get("custom_section", "parameter")
```

This configuration system makes the particle tagging codebase much more maintainable and portable across different computing environments.