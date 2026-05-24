# Darktag

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python package for assigning stellar mass to dark matter particles in dark matter-only simulations. Reproduces the sizes and stellar mass distributions of dwarf galaxies using advanced particle tagging methods.

## Features

- **Angular Momentum Tagging**: Assigns stellar mass to dark matter particles ranked by angular momentum
- **Spatial Tagging**: Distributes stellar mass via a Plummer profile
- **Binding Energy Tagging**: Assigns stellar mass to particles ranked by binding energy
- **JSON Configuration**: Centralized path and parameter management
- **AHF/HOP Halo Catalog Support**

## Installation

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Numerical computations |
| pandas | >=1.3.0 | Data manipulation |
| pynbody | >=2.1 | Simulation snapshot I/O |
| tangos | >=1.10.0 | Merger tree database |
| darklight | git+ | Stellar mass growth histories |
| scipy | >=1.7.0 | Scientific computing |
| scikit-learn | >=0.24.0 | ML utilities |
| matplotlib | >=3.5.0 | Plotting |
| seaborn | >=0.11.0 | Statistical viz |

```bash
pip install darktag
pip install darklight @ git+https://github.com/stacykim/darklight.git@main#egg=darklight
```

**Important:** The `darklight` package on PyPI is a different package. Always install from Git.

## Configuration

### config.json

Set paths and parameters in `config.json` at the package root:

```json
{
    "paths": {
        "tangos_path": "/path/to/tangos/databases/",
        "pynbody_path": "/path/to/simulation/data/",
        "manual_halonum_path": "",
        "manual_mstar_path": ""
    },
    "tagging": {
        "method": "angular_momentum",
        "ftag": 0.01,
        "clustering": {
            "method": "dbscan",
            "features": ["x", "y"],
            "scale": false
        }
    },
    "dbscan": {
        "eps": 0.05,
        "min_samples": 2
    },
    "hdbscan": {
        "min_cluster_size": 10,
        "min_samples": null,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom"
    },
    "darklight": {
        "n": 500,
        "DMO_OR_HYDRO": "DMO",
        "poccupied": "all"
    }
}
```

| Key | Description |
|-----|-------------|
| `paths.tangos_path` | Directory containing `.db` tangos database files |
| `paths.pynbody_path` | Directory containing simulation snapshot directories |
| `paths.manual_halonum_path` | Optional CSV for AHF halo number cross-references |
| `paths.manual_mstar_path` | Optional manual stellar mass path |
| `tagging.method` | Default tagging method |
| `tagging.ftag` | Tagging fraction free parameter (default: 0.01) |
| `tagging.clustering.method` | Clustering algorithm: `"dbscan"` or `"hdbscan"` |
| `tagging.clustering.features` | Feature columns for clustering, e.g. `["x","y"]`, `["x","y","z"]`, or `["x","y","z","vx","vy","vz"]` |
| `tagging.clustering.scale` | Whether to standardize features before clustering (`true`/`false`) |
| `dbscan.eps` | DBSCAN neighbourhood radius |
| `dbscan.min_samples` | DBSCAN minimum points per cluster |
| `hdbscan.min_cluster_size` | HDBSCAN minimum points per cluster |
| `hdbscan.min_samples` | HDBSCAN conservativeness (`null` = same as min_cluster_size) |
| `hdbscan.cluster_selection_epsilon` | Max merge distance for HDBSCAN (0.0 = pure hierarchical) |
| `hdbscan.cluster_selection_method` | `"eom"` (balanced) or `"leaf"` (fine-grained) |
| `hdbscan.allow_single_cluster` | Allow all particles in one cluster (`true`/`false`) |
| `hdbscan.max_cluster_size` | Max points per cluster (`null` = no limit) |
| `darklight.n` | Number of darklight Monte Carlo realizations (default: 500) |
| `darklight.DMO_OR_HYDRO` | `"DMO"` or `"HYDRO"` |
| `darklight.poccupied` | Occupation regime: `"all"`, `"nadler20"`, `"edge1"`, `"edgert"` |

### Accessing config

```python
from darktag.config import config

config.get_path('tangos_path')                      # path string
config.get('tagging', 'ftag')                        # 0.01
config.get_all_paths()                                # all path key-value pairs
```

**Override config file:** set the `DARKTAG_CONFIG` environment variable:
```bash
export DARKTAG_CONFIG=/custom/path/config.json
```

**Reload at runtime:**
```python
config.reload('/another/config.json')
```

## Usage

### 1. Tag particles

```python
import tangos
import darktag as dtag

tangos.core.init_db('/path/to/simulation.db')
sim = tangos.get_simulation('Halo1459_DMO')

df = dtag.tag_particles(
    DMO_database=sim,
    tagging_method='angular momentum',     # see methods below
    free_param_val=0.001,                   # tagging fraction
    include_mergers=True,                    # include accreted halos
    halonumber=1,                            # halo to analyze
    path_to_particle_data=None               # overrides config pynbody_path
)
```

Returns a DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `iords` | int64 | Particle IDs |
| `mstar` | float64 | Tagged stellar mass per particle per snapshot (Msun) |
| `t` | float64 | Time (Gyr) |
| `z` | float64 | Redshift |
| `type` | str | `"insitu"` (main halo) or `"accreted"` (merging halo) |

### 2. Calculate half-mass radii

```python
df_reff = dtag.calculate_rhalf(sim, df)

# Full version with more options:
df_reff = dtag.calculate_reffs_over_full_sim(
    DMOsim=sim,
    particles_tagged=df,
    pynbody_path=None,
    path_AHF_halonums=None
)
```

### Available tagging methods

| Method string | Function | Description |
|---|---|---|
| `'angular momentum'` | `angmom_tag_over_full_sim()` | Tags lowest-AM particles; non-recursive |
| `'angular momentum recursive'` | `angmom_tag_over_full_sim_recursive()` | Tags lowest-AM particles; recursive down merger tree |
| `'spatial'` | `spatial_tag_over_full_sim()` | Distributes mass via Plummer profile |

> **Binding energy** tagging (`BE_tag_over_full_sim`) is available as a direct function call but is not yet wired into the `tag_particles` dispatcher.

### EDGE simulation wrappers

```python
from darktag.edge.angular_momentum_tagging import angmom_tag_particles_edge

df = angmom_tag_particles_edge(
    sim_name='Halo1459_DMO',
    tagging_method='angular momentum',
    ftag_val=0.001,
    rec=False
)
```

Other EDGE wrappers: `BE_tag_particles_edge`, `spatial_tag_particles`, `angmom_calculate_reffs`.

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover the `Config` class and pure-logic utility functions. Tests for functions requiring `pynbody`/`tangos`/`darklight` are skipped when those packages are unavailable.

## Example workflow

```python
import tangos
import darktag as dtag
from darktag.config import config

# 1. Check paths
print(config.get_path('tangos_path'))
print(config.get_path('pynbody_path'))

# 2. Init simulation
tangos.core.init_db(config.get_path('tangos_path') + 'Halo1459.db')
sim = tangos.get_simulation('Halo1459_DMO')

# 3. Tag particles
df = dtag.tag_particles(sim, tagging_method='angular momentum', free_param_val=0.001)

# 4. Half-mass radii
df_reff = dtag.calculate_rhalf(sim, df)

# 5. Inspect results
print(df.head())
print(f"Tagged {len(df)} particle-snapshot rows")
```
