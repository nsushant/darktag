"""
Multi-instance HYDRO DM tagging using simulation stellar masses (integrate_sfr).

Unlike the DarkLight variant, stellar mass is computed deterministically from the
SFR histogram stored in tangos — no stochasticity per instance.

Usage:
    python scripts/run_tag_hydro_mstars.py Halo1459_HYDRO --n-instances 50
    python scripts/run_tag_hydro_mstars.py Halo1459_HYDRO --n-instances 50 --ftag 0.01
    python scripts/run_tag_hydro_mstars.py Halo1459_HYDRO_Mreionx02 --n-instances 50 --db-name Halo1459
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.angular_momentum_tagging import angmom_tag_multi_instance_hydro_mstars


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-instance HYDRO DM tagging using simulation stellar masses'
    )
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_HYDRO)')
    parser.add_argument('--n-instances', type=int, required=True,
                        help='Number of independent realisations')
    parser.add_argument('--ftag', type=float, default=0.01,
                        help='Tagging fraction (default: 0.01)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    parser.add_argument('--no-mergers', action='store_true', default=False,
                        help='Disable merger/accreted tagging')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output filenames')
    parser.add_argument('--db-name', type=str, default=None,
                        help='Tangos DB filename stem (default: first _-delimited token of sim_name, e.g. Halo1459)')
    parser.add_argument('--track-cluster-file', type=str, default=None,
                        help='track_cluster HDF5 file; switches to AHF catalogue and uses its halonums + cluster iords')
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')
    db_stem     = args.db_name or sim_name.split('_')[0]

    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    sim = tangos.get_simulation(sim_name)

    output_prefix = args.output_prefix or f'{sim_name}_tagged_hydro_mstars'
    filenames = angmom_tag_multi_instance_hydro_mstars(
        sim,
        n_instances=args.n_instances,
        halonumber=args.halonumber,
        free_param_value=args.ftag,
        output_prefix=output_prefix,
        track_cluster_file=args.track_cluster_file,
    )

    print(f'\nSaved {len(filenames)} files:')
    for fn in filenames:
        print(f'  {fn}')


if __name__ == '__main__':
    main()
