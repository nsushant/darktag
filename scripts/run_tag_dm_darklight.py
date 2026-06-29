"""
Multi-instance DM tagging using DarkLight for DMO and HYDRO simulations.

Usage:
    python scripts/run_tag_dm_darklight.py Halo1459_DMO --sim-type dmo --n-instances 50
    python scripts/run_tag_dm_darklight.py Halo1459_HYDRO --sim-type hydro --n-instances 50
    python scripts/run_tag_dm_darklight.py Halo1459_DMO --sim-type dmo --n-instances 50 --ftag 0.01
    python scripts/run_tag_dm_darklight.py Halo1459_DMO_Mreionx02 --sim-type dmo --n-instances 50 --db-name Halo1459
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.angular_momentum_tagging import (
    angmom_tag_multi_instance,
    angmom_tag_multi_instance_hydro_dm,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-instance DM tagging using DarkLight (DMO or HYDRO)'
    )
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--sim-type', choices=['dmo', 'hydro'], required=True,
                        help='Simulation type: dmo or hydro')
    parser.add_argument('--n-instances', type=int, required=True,
                        help='Number of independent DarkLight realisations')
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
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')
    db_stem     = args.db_name or sim_name.split('_')[0]

    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    sim = tangos.get_simulation(sim_name)

    if args.sim_type == 'dmo':
        output_prefix = args.output_prefix or f'{sim_name}_tagged_dm_dmo'
        filenames = angmom_tag_multi_instance(
            sim,
            n_instances=args.n_instances,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
        )
    else:
        output_prefix = args.output_prefix or f'{sim_name}_tagged_dm_hydro'
        filenames = angmom_tag_multi_instance_hydro_dm(
            sim,
            n_instances=args.n_instances,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
        )

    print(f'\nSaved {len(filenames)} files:')
    for fn in filenames:
        print(f'  {fn}')


if __name__ == '__main__':
    main()
