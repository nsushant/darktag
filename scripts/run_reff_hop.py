"""
Calculate half-mass radii (reff) for HOP-tagged particles.

Usage:
    python scripts/run_reff_hop.py Halo1459_DMO --ftag 0.01
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import calculate_reffs_over_full_sim


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Half-mass radii for HOP-tagged particles')
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    args = parser.parse_args()

    sim_name = args.sim_name
    tangos_path = config.get_path('tangos_path')
    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    import tangos
    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    tagged_csv = f'{sim_name}_tagged_hop.csv'
    if not os.path.isfile(tagged_csv):
        print(f'Error: {tagged_csv} not found. Run run_tagging_hop.py first.')
        sys.exit(1)

    reffs_fname = f'{sim_name}_reffs_hop.csv'

    df_reff = calculate_reffs_over_full_sim(
        DMOsim=sim,
        particles_tagged=tagged_csv,
        pynbody_path=pynbody_path,
        halo_number=args.halonumber,
        reffs_fname=reffs_fname,
    )

    print(f'\nSaved: {reffs_fname}')


if __name__ == '__main__':
    main()
