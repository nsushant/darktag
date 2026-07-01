"""
Calculate half-light and half-mass radii directly from HYDRO stellar particles.

No tagging required — deterministic single output CSV.
Applies edge_tangos_properties metallicity corrections, then uses
pynbody.analysis.luminosity.half_light_r for the halflight radius.
Voxel clustering isolates the main galaxy from satellites inside r200c.

Usage:
    python scripts/run_reff_hydro_stars.py Halo1459_HYDRO_Mreionx02
    python scripts/run_reff_hydro_stars.py Halo1459_HYDRO_Mreionx02 --voxel-size 0.08 --max-degree 20
    python scripts/run_reff_hydro_stars.py Halo1459_HYDRO_Mreionx02 --db-name Halo1459
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import calculate_reffs_hydro_stars


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Half-light / half-mass radii from HYDRO stellar particles'
    )
    parser.add_argument('sim_name',
                        help='Tangos HYDRO simulation name (e.g. Halo1459_HYDRO_Mreionx02)')
    parser.add_argument('--halonumber', type=int, default=0,
                        help='Halo index (default: 0)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV path (default: <sim_name>_hydro_reffs.csv)')
    parser.add_argument('--no-clustering', action='store_true', default=False,
                        help='Disable voxel clustering')
    parser.add_argument('--voxel-size', type=float, default=0.08,
                        help='Voxel edge length in kpc (default: 0.08)')
    parser.add_argument('--max-degree', type=int, default=20,
                        help='Max voxel connectivity radius in steps (default: 20)')
    parser.add_argument('--size-jump', type=float, default=2.0,
                        help='Cluster size ratio signalling satellite absorption (default: 2.0)')
    parser.add_argument('--ahf', action='store_true', default=False,
                        help='Use AHF halo catalogue instead of HOP')
    parser.add_argument('--db-name', type=str, default=None,
                        help='Tangos DB filename stem (default: first _-delimited token of sim_name)')
    parser.add_argument('--track-cluster-file', type=str, default=None,
                        help='track_cluster HDF5 file; switches to AHF catalogue and uses its halonums')
    args = parser.parse_args()

    sim_name     = args.sim_name
    tangos_path  = config.get_path('tangos_path')
    pynbody_path = config.get_path('pynbody_path')
    db_stem      = args.db_name or sim_name.split('_')[0]
    output_csv   = args.output_csv or f'{sim_name}_hydro_reffs.csv'

    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    sim = tangos.get_simulation(sim_name)

    df = calculate_reffs_hydro_stars(
        HYDROsim=sim,
        pynbody_path=pjoin(pynbody_path, sim_name),
        halo_number=args.halonumber,
        output_fname=output_csv,
        use_clustering=not args.no_clustering,
        use_ahf=args.ahf,
        voxel_size_kpc=args.voxel_size,
        max_degree=args.max_degree,
        size_jump=args.size_jump,
        track_cluster_file=args.track_cluster_file,
    )

    print(f'\nDone. Wrote {len(df)} snapshots to {output_csv}')


if __name__ == '__main__':
    main()
