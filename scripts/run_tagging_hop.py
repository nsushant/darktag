"""
HOP-based angular momentum tagging.

Usage:
    python scripts/run_tagging_hop.py Halo1459_DMO --ftag 0.01
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.expanduser('~'))

from config import config
from darktag.tagging.angular_momentum_tagging_hydrodynamic_sim import angmom_tag_over_full_sim_recursive


def main():
    import argparse

    parser = argparse.ArgumentParser(description='HOP-based angular momentum tagging')
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--ftag', type=float, default=0.01,
                        help='Tagging fraction (default: 0.01)')
    parser.add_argument('--mergers', action='store_true', default=True,
                        help='Include mergers (default: True)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    args = parser.parse_args()

    sim_name = args.sim_name
    tangos_path = config.get_path('tangos_path')
    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    import tangos
    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    df, _ = angmom_tag_over_full_sim_recursive(
        sim, -1, args.halonumber,
        free_param_value=args.ftag,
        pynbody_path=pynbody_path,
        mergers=args.mergers,
    )

    output = f'{sim_name}_tagged_hop.csv'
    df.to_csv(output, index=False)
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
