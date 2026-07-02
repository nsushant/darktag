"""
Compute stellar half-light and half-mass radii from a track_cluster HDF5 file.

For each snapshot in the HDF5 (main branch only by default):
  - Reads DM cluster centroid, bounding_radius, and AHF halo number
  - Loads the HYDRO snapshot
  - Applies etp metallicity corrections
  - Centres using the AHF halo
  - Selects stars within bounding_radius of centroid
  - Optionally runs voxel clustering on those stars to exclude stellar satellites
  - Computes half-light (pynbody SSP) and projected half-mass
  - Writes incremental CSV

Usage:
    python scripts/run_halflight_from_track.py Halo1459_HYDRO_Mreionx02 \\
        --hdf5 Halo1459_HYDRO_Mreionx02_cluster_tree.hdf5
"""

import sys
import os
import argparse
import gc
from os.path import join as pjoin

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import _voxel_pick_cluster


def main():
    parser = argparse.ArgumentParser(
        description='Stellar halflight from track_cluster HDF5'
    )
    parser.add_argument('sim_name',
                        help='HYDRO simulation name (e.g. Halo1459_HYDRO_Mreionx02)')
    parser.add_argument('--hdf5', default=None,
                        help='Path to track_cluster HDF5 (default: <sim_name>_cluster_tree.hdf5)')
    parser.add_argument('--branch', default='main',
                        help='Branch to process from HDF5 (default: main)')
    parser.add_argument('--output-csv', default=None,
                        help='Output CSV path (default: <sim_name>_halflight_track.csv)')
    parser.add_argument('--no-clustering', action='store_true',
                        help='Skip stellar voxel clustering (use all stars in bounding sphere)')
    parser.add_argument('--voxel-size', type=float, default=0.08,
                        help='Stellar voxel size in kpc (default: 0.08)')
    parser.add_argument('--max-degree', type=int, default=20,
                        help='Max voxel connectivity steps (default: 20)')
    parser.add_argument('--size-jump', type=float, default=2.0,
                        help='Cluster size jump threshold (default: 2.0)')
    parser.add_argument('--radius-buffer', type=float, default=1.2,
                        help='Multiplier on DM bounding_radius for star selection (default: 1.2)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    hdf5_path   = args.hdf5   or f'{sim_name}_cluster_tree.hdf5'
    branch      = args.branch
    output_csv  = args.output_csv or f'{sim_name}_halflight_track.csv'
    use_cluster = not args.no_clustering
    voxel_size  = args.voxel_size
    max_degree  = args.max_degree
    size_jump   = args.size_jump
    rad_buf     = args.radius_buffer

    pynbody_path = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')

    try:
        import pynbody
        import pynbody.analysis.luminosity
        import h5py
    except ImportError as e:
        print(f'Missing dependency: {e}')
        sys.exit(1)

    try:
        import edge_tangos_properties as etp
    except ImportError:
        etp = None
        print('Warning: edge_tangos_properties not found — metallicity corrections skipped')

    if not os.path.isfile(hdf5_path):
        print(f'HDF5 not found: {hdf5_path}')
        sys.exit(1)

    with h5py.File(hdf5_path, 'r') as f:
        snapshots = sorted(f.keys())   # output_0001, output_0002, ...
        snap_data = {}
        for snap in snapshots:
            if branch not in f[snap]:
                continue
            grp = f[snap][branch]
            if 'centroid' not in grp or 'bounding_radius' not in grp:
                print(f'  {snap}: missing centroid/bounding_radius, skipping')
                continue
            snap_data[snap] = {
                'halonum':        int(grp['halonum'][()]),
                'centroid':       grp['centroid'][:],
                'bounding_radius': float(grp['bounding_radius'][()]),
            }

    if not snap_data:
        print(f'No usable snapshots found in {hdf5_path} for branch "{branch}"')
        sys.exit(1)

    print(f'{len(snap_data)} snapshots to process')

    # Resume
    processed = set()
    stored = {'halflight': [], 'reff': [], 'z': [], 't': []}

    if os.path.isfile(output_csv):
        try:
            existing = pd.read_csv(output_csv, index_col=0)
            if len(existing) > 0:
                # use snapshot name stored in 'output' column if present, else skip resume
                if 'output' in existing.columns:
                    processed = set(existing['output'].values)
                stored['halflight'] = existing['halflight'].tolist()
                stored['reff']      = existing['reff'].tolist()
                stored['z']         = existing['z'].tolist()
                stored['t']         = existing['t'].tolist()
                if 'output' in existing.columns:
                    stored['output'] = existing['output'].tolist()
                print(f'Resuming: {len(existing)} snapshots already done')
        except Exception as e:
            print(f'Could not read existing CSV ({e}), starting fresh')

    if 'output' not in stored:
        stored['output'] = []

    prev_st_iords = np.array([])

    # Process z=0 first (reversed)
    for snap in reversed(list(snap_data.keys())):
        if snap in processed:
            print(f'Skipping {snap} (already done)')
            continue

        info      = snap_data[snap]
        halonum   = info['halonum']
        centroid  = info['centroid']
        bound_r   = info['bounding_radius'] * rad_buf

        print(f'\n── {snap} (halo {halonum}, bounding_r={bound_r/rad_buf:.2f}×{rad_buf}={bound_r:.2f} kpc) ──')

        simfn = pjoin(pynbody_path, snap)
        try:
            HYDROparticles = pynbody.load(simfn)
        except Exception as e:
            print(f'  Failed to load snapshot: {e}, skipping')
            continue

        # etp metallicity corrections on full snap before any filtering
        if etp is not None:
            try:
                etp.stars.StellarProperty._ensure_ramses_metal_are_corrected(HYDROparticles)
            except Exception as e:
                print(f'  etp correction failed ({e}), continuing without')

        try:
            pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
            halo_cat = HYDROparticles.halos(halo_numbers='v1')
            h = halo_cat[halonum]
            HYDROparticles.physical_units()
            pynbody.analysis.halo.center(h)
        except Exception as e:
            print(f'  Centering failed: {e}, skipping')
            del HYDROparticles
            continue

        try:
            t_val = float(HYDROparticles.properties.get('time', np.nan))
            z_val = float(HYDROparticles.properties.get('z', np.nan))
        except Exception:
            t_val, z_val = np.nan, np.nan

        # Select stars within buffered DM bounding radius of DM centroid
        # centroid is already in the centred frame (track_cluster centres before clustering)
        st_pos  = np.array(HYDROparticles.st['pos'])
        st_dist = np.sqrt(((st_pos - centroid)**2).sum(axis=1))
        stars   = HYDROparticles.st[st_dist <= bound_r]

        if len(stars) == 0:
            print('  No stars within DM bounding radius, skipping')
            del HYDROparticles
            continue

        # Filter zero-iron-metallicity stars
        if etp is not None:
            try:
                good = etp.stars.AbundanceRatios._mask_stars_with_zero_iron_metallicity(stars)
                stars = stars[good]
            except Exception as e:
                print(f'  zero-iron mask failed ({e}), skipping filter')

        if len(stars) == 0:
            print('  No stars after metallicity filter, skipping')
            del HYDROparticles
            continue

        # Voxel cluster stars to exclude stellar satellites
        if use_cluster:
            st_positions = np.array(stars['pos'])
            st_iords     = np.asarray(stars['iord'])
            mask_st = _voxel_pick_cluster(
                st_positions, st_iords, voxel_size,
                prev_iords=prev_st_iords if len(prev_st_iords) > 0 else None,
                max_degree=max_degree,
                size_jump=size_jump,
            )
            if mask_st is None or mask_st.sum() == 0:
                print('  Stellar voxel clustering returned empty cluster, skipping')
                del HYDROparticles
                continue
            cluster_stars  = stars[mask_st]
            prev_st_iords  = np.asarray(cluster_stars['iord'])
        else:
            cluster_stars = stars
            prev_st_iords = np.asarray(cluster_stars['iord'])

        # Half-light radius
        try:
            hlight = float(pynbody.analysis.luminosity.half_light_r(cluster_stars, band='V'))
        except Exception as e:
            print(f'  half_light_r failed ({e}), storing NaN')
            hlight = float('nan')

        # Projected half-mass (cylindrical, face-on already from centre)
        try:
            pynbody.analysis.angmom.faceon(cluster_stars)
        except Exception:
            pass
        rxy          = np.sqrt(np.array(cluster_stars['x'])**2 + np.array(cluster_stars['y'])**2)
        sorted_idx   = np.argsort(rxy)
        sorted_mass  = np.array(cluster_stars['mass'])[sorted_idx]
        cumsum_mass  = np.cumsum(sorted_mass)
        R_half       = float(rxy[sorted_idx][np.where(cumsum_mass >= cumsum_mass[-1] / 2)[0][0]])

        stored['halflight'].append(hlight)
        stored['reff'].append(R_half)
        stored['z'].append(z_val)
        stored['t'].append(t_val)
        stored['output'].append(snap)

        df_out = pd.DataFrame(stored)
        df_out.to_csv(output_csv)
        print(f'  halflight={hlight:.4f} kpc  reff={R_half:.4f} kpc  → wrote {output_csv}')

        del HYDROparticles
        gc.collect()

    print(f'\nDone. Results in {output_csv}')


if __name__ == '__main__':
    main()
