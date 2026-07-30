"""
AHF -> HOP conversion.

Reads one or more AHF track files produced by scripts/track_ahf.py
(e.g. Halo1459_DMO_Mreionx12_ahf_track.hdf5) and, for every snapshot, finds the
HOP halo that the tracked AHF halo corresponds to — the HOP halo whose DM iords
overlap most with the AHF halo's (child-removed) iords.

For each input <sim>_ahf_track.hdf5 it writes <sim>_hop_track.hdf5 with the same
per-snapshot structure plus the matched HOP halo number:
    /<snap>/main/halonum       int64   AHF (v1) halo number      (copied)
    /<snap>/main/hop_halonum   int64   matched HOP halo (1-based)
    /<snap>/main/iords         int64[] child-removed DM iords    (copied)

The 1-based hop_halonum matches the convention used elsewhere (index the pynbody
HOP catalogue with hop_halonum - 1). The reff / tagging readers prefer
hop_halonum when present, so the output is a drop-in HOP-based track file.

Usage:
    python scripts/ahf_to_hop_conversion.py Halo1459_DMO_Mreionx12_ahf_track.hdf5 --dmo
    python scripts/ahf_to_hop_conversion.py *_ahf_track.hdf5 --dmo
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


def find_hop_halonum(hop_cat, target_iords, max_halos=None):
    """Return (hop_halonum_1based, overlap, n_target) for the HOP halo with the
    most DM iord overlap with target_iords. hop_halonum is None if no overlap.

    Stops early once a halo covers more than half of target_iords.
    """
    target = np.unique(np.asarray(target_iords))
    n_target = len(target)
    try:
        n_halos = len(hop_cat)
    except Exception:
        n_halos = 0
    limit = n_halos if max_halos is None else min(max_halos, n_halos)

    best_num, best_ov = None, 0
    for idx in range(limit):            # HOP catalogue is 0-indexed
        try:
            h = hop_cat[idx]
            ov = int(np.isin(np.asarray(h.dm['iord']), target).sum())
        except Exception:
            continue
        if ov > best_ov:
            best_ov = ov
            best_num = idx + 1          # store 1-based
            if best_ov > 0.5 * n_target:
                break

    return best_num, best_ov, n_target


def convert_file(track_path, pynbody_path, max_halos=None):
    """Read one AHF track file, match HOP halos per snapshot, write *_hop_track.hdf5."""
    import h5py
    import pynbody
    import pynbody.halo.hop

    base = os.path.basename(track_path)
    if base.endswith('_ahf_track.hdf5'):
        out_path = track_path[:-len('_ahf_track.hdf5')] + '_hop_track.hdf5'
    elif track_path.endswith('.hdf5'):
        out_path = track_path[:-len('.hdf5')] + '_hop.hdf5'
    else:
        out_path = track_path + '_hop.hdf5'

    print(f'\n=== {base} -> {os.path.basename(out_path)} ===')

    with h5py.File(track_path, 'r') as fin:
        snaps = [s for s in fin.keys() if 'main' in fin[s] and 'iords' in fin[s]['main']]
        snaps.sort()

        with h5py.File(out_path, 'w') as fout:
            for snap in snaps:
                mg_in = fin[snap]['main']
                ahf_num = int(mg_in['halonum'][()]) if 'halonum' in mg_in else -1
                iords   = mg_in['iords'][:]

                simfn = pjoin(pynbody_path, snap)
                try:
                    s = pynbody.load(simfn)
                    s.physical_units()
                except Exception as e:
                    print(f'  {snap}: failed to load snapshot ({e}), skipping')
                    continue
                try:
                    pynbody.config['halo-class-priority'] = [pynbody.halo.hop.HOPCatalogue]
                    hop_cat = s.halos()
                except Exception as e:
                    print(f'  {snap}: failed to load HOP catalogue ({e}), skipping')
                    del s
                    continue

                hop_num, ov, n = find_hop_halonum(hop_cat, iords, max_halos=max_halos)
                if hop_num is None:
                    print(f'  {snap}: AHF {ahf_num} -> no HOP match (0/{n}), skipping')
                    del s
                    continue

                grp = fout.require_group(snap).require_group('main')
                grp.create_dataset('halonum',     data=np.int64(ahf_num))
                grp.create_dataset('hop_halonum', data=np.int64(hop_num))
                grp.create_dataset('iords',       data=np.asarray(iords).astype(np.int64))
                fout.flush()

                print(f'  {snap}: AHF {ahf_num} -> HOP {hop_num} (overlap {ov}/{n})')
                del s

    print(f'  wrote {out_path}')
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Match AHF-tracked halos to HOP halos per snapshot')
    parser.add_argument('track_files', nargs='+',
                        help='One or more *_ahf_track.hdf5 files (globs allowed)')
    parser.add_argument('--dmo', action='store_true',
                        help='Use the DMO pynbody_path from config (default: hydro path)')
    parser.add_argument('--max-halos', type=int, default=None,
                        help='Only search the first N HOP halos per snapshot (default: all)')
    args = parser.parse_args()

    if args.dmo:
        base_path = config.get_path('pynbody_path')
    else:
        base_path = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')

    # Expand any globs the shell didn't
    track_files = []
    for pat in args.track_files:
        matches = sorted(glob.glob(pat))
        track_files.extend(matches if matches else [pat])

    if not track_files:
        print('Error: no track files given')
        sys.exit(1)

    for track_path in track_files:
        if not os.path.isfile(track_path):
            print(f'Skipping {track_path}: not found')
            continue
        # Derive the sim name (folder holding the snapshots) from the filename
        stem = os.path.basename(track_path)
        for suffix in ('_ahf_track.hdf5', '_hop_track.hdf5', '.hdf5'):
            if stem.endswith(suffix):
                sim_name = stem[:-len(suffix)]
                break
        else:
            sim_name = stem
        pynbody_path = pjoin(base_path, sim_name)
        print(f'\nSim: {sim_name}  (snapshots under {pynbody_path})')
        convert_file(track_path, pynbody_path, max_halos=args.max_halos)

    print('\nDone.')


if __name__ == '__main__':
    main()
