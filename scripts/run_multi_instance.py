"""
Multi-instance angular momentum tagging.

Runs N independent DarkLight realisations (each with n=1) over a simulation,
loading each snapshot once and writing one output CSV per instance.

Usage (after pip install -e . from repo root):
    # Non-recursive (default):
    python scripts/run_multi_instance.py Halo1459_DMO --n-instances 50
    python scripts/run_multi_instance.py Halo1459_DMO --n-instances 50 --ftag 0.01

    # Recursive (walks full merger tree):
    python scripts/run_multi_instance.py Halo1459_DMO --n-instances 50 --recursive
    python scripts/run_multi_instance.py Halo1459_DMO --n-instances 50 --recursive --output-prefix my_run
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.angular_momentum_tagging import (
    angmom_tag_multi_instance,
    angmom_tag_multi_instance_recursive,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-instance angular momentum tagging')
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--n-instances', type=int, required=True,
                        help='Number of independent DarkLight realisations')
    parser.add_argument('--ftag', type=float, default=0.01,
                        help='Tagging fraction (default: 0.01)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    parser.add_argument('--no-mergers', action='store_true', default=False,
                        help='Disable merger/accreted tagging')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output filenames (default: {sim_name}_tagged[_recursive])')
    parser.add_argument('--occupation-frac', type=str, default='all',
                        help="Occupation fraction regime: 'all', 'nadler20', 'edge1', 'edgert' (default: all)")
    parser.add_argument('--recursive', action='store_true', default=False,
                        help='Use recursive variant (walks full merger tree, mirrors run_tagging_hop.py)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')

    import tangos
    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    if args.recursive:
        output_prefix = args.output_prefix or f'{sim_name}_tagged_recursive'
        dfs, _ = angmom_tag_multi_instance_recursive(
            sim,
            n_instances=args.n_instances,
            tstep=-1,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
        )
        filenames = [os.path.join(output_prefix, f"instance_{k:03d}.csv") for k in range(args.n_instances)]
    else:
        output_prefix = args.output_prefix or f'{sim_name}_tagged'
        filenames = angmom_tag_multi_instance(
            sim,
            n_instances=args.n_instances,
            halonumber=args.halonumber,
            free_param_value=args.ftag,
            output_prefix=output_prefix,
            mergers=not args.no_mergers,
            occupation_frac=args.occupation_frac,
        )

    print(f'\nSaved {len(filenames)} files:')
    for fn in filenames:
        print(f'  {fn}')


if __name__ == '__main__':
    main()
