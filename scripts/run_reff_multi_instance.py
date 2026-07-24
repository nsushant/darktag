"""
Calculate half-mass / half-light radii for multi-instance tagged particle files.

Loads each snapshot once and computes reffs for all instance CSVs in the tagged
directory, with each instance tracking its own DBSCAN cluster independently.

Usage:
    python scripts/run_reff_multi_instance.py Halo1459_DMO --tagged-dir Halo1459_DMO_tagged
    python scripts/run_reff_multi_instance.py Halo1459_DMO --tagged-dir Halo1459_DMO_tagged --output-dir Halo1459_DMO_reffs
"""

import sys
import os
from os.path import join as pjoin

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import calculate_reffs_multi_instance


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reff calculation for multi-instance tagging output')
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--tagged-dir', required=True,
                        help='Directory containing instance_*.csv tagged particle files')
    parser.add_argument('--halonumber', type=int, default=0,
                        help='Halo number (default: 0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output reff CSVs (default: <tagged-dir>_reffs)')
    parser.add_argument('--no-clustering', action='store_true', default=False,
                        help='Disable density region growing when calculating reffs')
    parser.add_argument('--voxel-size', type=float, default=0.08,
                        help='Voxel edge length in kpc for region growing (default: 0.08)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of density peaks to try for region growing (default: 3)')
    parser.add_argument('--min-shells', type=int, default=3,
                        help='Minimum shells before allowing a density cut (default: 3)')
    parser.add_argument('--ahf', action='store_true', default=False,
                        help='Use AHF halo catalogue instead of HOP')
    parser.add_argument('--track-cluster-file', type=str, default=None,
                        help='track_cluster HDF5 file; switches to AHF catalogue and uses its halonums')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Only process the first N instance files (useful for testing)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    tangos_path = config.get_path('tangos_path')
    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    import tangos
    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    # Accept a single CSV file — copy it into a temp dir as instance_000.csv
    tagged_dir = args.tagged_dir
    _tmp_dir   = None
    if os.path.isfile(tagged_dir) and tagged_dir.endswith('.csv'):
        import tempfile, shutil
        _tmp_dir   = tempfile.mkdtemp()
        shutil.copy(tagged_dir, os.path.join(_tmp_dir, 'instance_000.csv'))
        tagged_dir = _tmp_dir
    elif not os.path.isdir(tagged_dir):
        print(f'Error: {tagged_dir} is not a directory or CSV file.')
        sys.exit(1)

    dfs = calculate_reffs_multi_instance(
        DMOsim=sim,
        tagged_dir=tagged_dir,
        pynbody_path=pynbody_path,
        halo_number=args.halonumber,
        output_dir=args.output_dir,
        use_clustering=not args.no_clustering,
        use_ahf=args.ahf,
        track_cluster_file=args.track_cluster_file,
        max_instances=args.max_instances,
        voxel_size_kpc=args.voxel_size,
        n_seeds=args.n_seeds,
        min_shells=args.min_shells,
    )

    out_dir = args.output_dir or args.tagged_dir.rstrip('/') + '_reffs'
    print(f'\nDone. Wrote {len(dfs)} reff files to {out_dir}/')


if __name__ == '__main__':
    main()
