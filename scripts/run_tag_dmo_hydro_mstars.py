"""
Tag DMO DM particles using stellar mass from the paired HYDRO simulation
(SFR_histogram via integrate_sfr). No DarkLight stochasticity.

Usage:
    python scripts/run_tag_dmo_hydro_mstars.py Halo1459_DMO \
        --hydro-sim-name Halo1459_HYDRO \
        --n-instances 1 --ftag 0.01 --db-name Halo1459

    python scripts/run_tag_dmo_hydro_mstars.py Halo1459_DMO_Mreionx02 \
        --hydro-sim-name Halo1459_fiducial_Mreionx02 \
        --n-instances 1 --ftag 0.01 --db-name Halo1459 \
        --track-cluster-file Halo1459_DMO_Mreionx02_hop_track.hdf5
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.angular_momentum_tagging import angmom_tag_dmo_hydro_mstars


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Tag DMO DM particles using HYDRO stellar mass (SFR_histogram)'
    )
    parser.add_argument('sim_name', help='Tangos DMO simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--hydro-sim-name', required=True,
                        help='Tangos HYDRO simulation name (e.g. Halo1459_HYDRO)')
    parser.add_argument('--n-instances', type=int, required=True,
                        help='Number of output instances (results are identical; n=1 is fine)')
    parser.add_argument('--ftag', type=float, default=0.01,
                        help='Tagging fraction (default: 0.01)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo number (default: 1)')
    parser.add_argument('--no-mergers', action='store_true', default=False,
                        help='Disable merger/accreted tagging (DarkLight on merging halos)')
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Directory for output CSVs')
    parser.add_argument('--db-name', type=str, default=None,
                        help='Tangos DB filename stem (default: first _-delimited token of sim_name)')
    parser.add_argument('--track-cluster-file', type=str, default=None,
                        help='track_cluster HDF5 file; auto-detects hop_halonum (HOP) or halonum (AHF)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')
    db_stem     = args.db_name or sim_name.split('_')[0]

    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    dmo_sim   = tangos.get_simulation(sim_name)
    hydro_sim = tangos.get_simulation(args.hydro_sim_name)

    output_prefix = args.output_prefix or f'{sim_name}_tagged_dmo_hydromstars'
    filenames = angmom_tag_dmo_hydro_mstars(
        dmo_sim,
        hydro_sim,
        n_instances=args.n_instances,
        halonumber=args.halonumber,
        free_param_value=args.ftag,
        output_prefix=output_prefix,
        track_cluster_file=args.track_cluster_file,
        mergers=not args.no_mergers,
    )

    print(f'\nSaved {len(filenames)} files:')
    for fn in filenames:
        print(f'  {fn}')


if __name__ == '__main__':
    main()
