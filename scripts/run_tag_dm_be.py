"""
Multi-instance DM tagging using pre-computed binding energy cache.

Usage:
    python scripts/run_tag_dm_be.py Halo1459_DMO --sim-type dmo --be-cache Halo1459_DMO_binding_energies.hdf5 --n-instances 50
    python scripts/run_tag_dm_be.py Halo1459_HYDRO --sim-type hydro --be-cache Halo1459_HYDRO_binding_energies.hdf5 --n-instances 50
    python scripts/run_tag_dm_be.py Halo1459_DMO_Mreionx02 --sim-type dmo --be-cache Halo1459_DMO_Mreionx02_binding_energies.hdf5 --track-cluster-file Halo1459_DMO_Mreionx02_cluster_tree.hdf5 --n-instances 50 --ftag 0.01
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.binding_energy_tagging import (
    be_tag_multi_instance,
    be_tag_multi_instance_hydro_dm,
    be_tag_multi_instance_hydro_mstars,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-instance DM tagging using pre-computed binding energy cache'
    )
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--sim-type', choices=['dmo', 'hydro', 'hydro-mstars'], required=True,
                        help='Simulation type: dmo, hydro (DarkLight), or hydro-mstars (direct stellar mass)')
    parser.add_argument('--be-cache', required=True,
                        help='Path to binding energy cache HDF5 file')
    parser.add_argument('--n-instances', type=int, required=True,
                        help='Number of independent DarkLight realisations')
    parser.add_argument('--ftag', type=float, default=0.02,
                        help='Tagging fraction (default: 0.01)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    parser.add_argument('--no-mergers', action='store_true', default=False,
                        help='Disable merger/accreted tagging')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix/directory for output filenames')
    parser.add_argument('--db-name', type=str, default=None,
                        help='Tangos DB filename stem (default: first _-delimited token of sim_name)')
    parser.add_argument('--track-cluster-file', type=str, default=None,
                        help='track_cluster HDF5 file; auto-detects hop_halonum (HOP) or halonum (AHF) and uses appropriate catalogue')
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')
    db_stem     = args.db_name or sim_name.split('_')[0]

    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    sim = tangos.get_simulation(sim_name)

    if args.sim_type == 'dmo':
        output_prefix = args.output_prefix or f'{sim_name}_tagged_dm_be_dmo'
        filenames = be_tag_multi_instance(
            sim,
            n_instances=args.n_instances,
            be_cache=args.be_cache,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
            track_cluster_file=args.track_cluster_file,
        )
    elif args.sim_type == 'hydro':
        output_prefix = args.output_prefix or f'{sim_name}_tagged_dm_be_hydro'
        filenames = be_tag_multi_instance_hydro_dm(
            sim,
            n_instances=args.n_instances,
            be_cache=args.be_cache,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
            track_cluster_file=args.track_cluster_file,
        )
    else:
        output_prefix = args.output_prefix or f'{sim_name}_tagged_dm_be_hydro_mstars'
        filenames = be_tag_multi_instance_hydro_mstars(
            sim,
            n_instances=args.n_instances,
            be_cache=args.be_cache,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
            track_cluster_file=args.track_cluster_file,
        )

    print(f'\nSaved {len(filenames)} files:')
    for fn in filenames:
        print(f'  {fn}')


if __name__ == '__main__':
    main()
