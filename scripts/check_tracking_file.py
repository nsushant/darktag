"""
Print a summary of entries in a track_cluster HDF5 file.

Usage:
    python scripts/check_tracking_file.py <hdf5_file>
"""

import sys
import h5py
import numpy as np


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/check_tracking_file.py <hdf5_file>')
        sys.exit(1)

    path = sys.argv[1]

    with h5py.File(path, 'r') as f:
        snaps = sorted(f.keys())
        n_snaps = len(snaps)

        branch_counts = {}
        halonum_reliable = {}

        for snap in snaps:
            grp = f[snap]
            for branch in grp.keys():
                branch_counts[branch] = branch_counts.get(branch, 0) + 1
                if 'halonum_reliable' in grp[branch]:
                    reliable = bool(grp[branch]['halonum_reliable'][()])
                    if branch not in halonum_reliable:
                        halonum_reliable[branch] = {'reliable': 0, 'carried': 0}
                    if reliable:
                        halonum_reliable[branch]['reliable'] += 1
                    else:
                        halonum_reliable[branch]['carried'] += 1

        print(f'File:       {path}')
        print(f'Snapshots:  {n_snaps}')
        if snaps:
            print(f'Range:      {snaps[0]}  →  {snaps[-1]}')
        print(f'Branches:   {len(branch_counts)}')
        print()

        for branch in sorted(branch_counts):
            count = branch_counts[branch]
            rel = halonum_reliable.get(branch, {})
            reliable_str = ''
            if rel:
                reliable_str = f'  (reliable: {rel["reliable"]}, carried: {rel["carried"]})'
            print(f'  {branch:<12}  {count} snaps{reliable_str}')


if __name__ == '__main__':
    main()
