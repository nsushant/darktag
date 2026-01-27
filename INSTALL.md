# Darktag Installation Guide

## PyPI Installation

### Basic Installation
```bash
pip install darktag
```

### Installation with Astrophysics Dependencies
```bash
pip install darktag[astrophysics]
```

### Manual Astrophysics Dependencies (if pip install fails)
If the optional dependencies fail to install, you can install them manually:

```bash
# Install from source repositories
pip install pynbody @ git+https://github.com/pynbody/pynbody.git
pip install tangos @ git+https://github.com/pynbody/tangos.git  
pip install darklight @ git+https://github.com/stacykim/darklight.git
```

## Verification

Test the installation:
```bash
python -c "import darktag; print('✓ Darktag package installed successfully!')"
```

## Usage

### Basic Usage (Core Dependencies Only)
```python
import darktag as dtag
from config import config

# Basic configuration and utils work
print(f"Package version: {dtag.__version__}")
```

### Full Usage (With Astrophysics Dependencies)
```python
import tangos
import darktag as dtag
from config import config

# Initialize tangos database
tangos.core.init_db('your_simulation.db')
DMO_database = tangos.get_simulation('your_simulation_name')

# Perform particle tagging
df_tagged_particles = dtag.tag_particles(
    DMO_database, 
    tagging_method='angular momentum',
    free_param_val=0.001
)

# Calculate half-mass radii
df_half_mass_tagged = dtag.calculate_rhalf(
    DMO_database, 
    df_tagged_particles
)
```

## Dependencies

### Core Dependencies (Always Required)
- numpy>=1.20.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scipy>=1.7.0
- scikit-learn>=0.24.0

### Optional Astrophysics Dependencies
- pynbody: Astrophysical simulation analysis
- tangos: Simulation database management  
- darklight: Stellar mass growth histories

## Troubleshooting

### Import Errors
If you see warnings about missing astrophysics dependencies:

```bash
# Install optional dependencies
pip install darktag[astrophysics]

# Or install manually from source
pip install pynbody @ git+https://github.com/pynbody/pynbody.git
pip install tangos @ git+https://github.com/pynbody/tangos.git
```

### Configuration
The package uses the same `config.json` file as before. Update paths in `config.json`:

```json
{
    "paths": {
        "tangos_path": "/path/to/your/tangos/databases/",
        "pynbody_path": "/path/to/your/pynbody/data/",
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