
# Darktrace Package

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python package for assigning stellar mass to dark matter particles in dark matter-only simulations. The primary goal is to accurately reproduce the sizes and stellar mass distributions of dwarf galaxies using advanced particle tagging methods.

<img width="1183" alt="image" src="https://github.com/nsushant/particle_tagging_package/assets/64201587/9cd0684d-7a8f-4015-b329-4456a1f3c27b">

*Figure 1. White circle shows the calculated halflight radius. Blue circle shows the virial radius of the dark matter halo and contours show the stellar mass distribution created by particle tagging.*

## Features

- **Angular Momentum Tagging**: Associates stellar mass with dark matter particles using their angular momenta
- **Spatial Tagging**: Distributes stellar mass across the galaxy based on a Plummer Profile
- **Binding Energy Tagging**: Associates stellar mass with dark matter particles using their binding energies
- **Centralized Configuration**: JSON-based configuration system for easy path management
- **HALO Catalog Support**: Compatible with AHF and HOP halo catalogues

## Installation

### Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn pynbody
```

**Note**: `tangos` and `darklight` are specialized astrophysics packages that may need to be installed from source or specific repositories.

### Install Darktrace

Since this package is not available on PyPI, install it directly from the repository:

1. Clone the repository:
```bash
git clone https://github.com/nsushantnigudkar/darktrace.git
cd darktrace
```

## Configuration

Before using the package, you need to configure the paths in `config/config.json`:

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

Update the paths to point to your simulation data and tangos databases.

## Usage

### Basic Usage

```python
import tangos
import darktrace as dtrace
from config import config

# Initialize tangos database
tangos.core.init_db('your_simulation.db')

# Load simulation database
DMO_database = tangos.get_simulation('your_simulation_name')

# Perform particle tagging
df_tagged_particles = dtrace.tag_particles(
    DMO_database, 
    path_to_particle_data=config.get_path("pynbody_path"),
    tagging_method='angular momentum',
    free_param_val=0.001
)

# Calculate half-mass radii
df_half_mass_tagged = dtrace.calculate_rhalf(
    DMO_database, 
    df_tagged_particles,
    pynbody_path=config.get_path("pynbody_path")
)
```

### Available Tagging Methods

1. **Angular Momentum**: `tagging_method='angular momentum'`
2. **Spatial**: `tagging_method='spatial'`
3. **Binding Energy**: `tagging_method='binding energy'`

### Advanced Configuration

The package supports various parameters for fine-tuning:

- `free_param_val`: Free parameter for tagging method (default: 0.01)
- `include_mergers`: Whether to include merger events (default: True)
- `halonumber`: Specific halo number to analyze (default: 1)

## Examples

See the `examples/` directory for complete tutorial scripts:

- `tutorial_1_getting_started.py`: Basic particle tagging workflow
- `tutorial_2_plotting_stellar_mass_distributions.py`: Plotting and analysis

Run the example:
```bash
cd examples
python tutorial_1_getting_started.py
```

## Dependencies

### Core Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **pynbody**: Astrophysical simulation analysis
- **tangos**: Simulation database management
- **darklight**: Halo analysis tools

### Optional Dependencies (for plotting)
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization

### Standard Library
- `os`, `csv`, `random` (included with Python)

## Data Requirements

The tagging methods require:
- **Tangos databases** with merger trees
- **Pynbody particle data** 
- **AHF or HOP halo catalogues**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{darktrace,
  title={Darktrace: Particle Tagging for Dark Matter Simulations},
  author={Nigudkar, Sushant},
  year={2024},
  url={https://github.com/nsushantnigudkar/darktrace}
}
```

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository.
