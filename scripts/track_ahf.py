"""
Simple AHF halo tracker.

Follows one halo back in time (z=0 -> high z) by matching the AHF halo with the
most DM iord overlap between consecutive snapshots. Before every overlap check a
halo's child/sub-halo DM particles are removed, so the match is driven by the
halo's own particles rather than its substructure.

With --prefer-stars, only halos that contain star particles are considered as
candidates (the overlap check is skipped for star-less halos up front).

No voxels, no tangos merger tree, no positions/r200 — just AHF membership and
iord overlap.

Output HDF5 (consumed directly by the reff / tagging readers via their AHF path):
    /<output>/main/halonum   int64   1-based AHF (v1) halo number
    /<output>/main/iords     int64[] child-removed DM iords

Usage:
    python scripts/track_ahf.py Halo1459_HYDRO_Mreionx02 --halonumber 1 --prefer-stars
    python scripts/track_ahf.py Halo1459_DMO --halonumber 1 --dmo
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


def _child_dm_iords(h, halo_cat, _seen=None):
    """Recursively gather the DM iords of every child/sub-halo of h.

    Reads the AHF 'children' property; returns an empty array if the halo has no
    children (or none can be loaded).
    """
    if _seen is None:
        _seen = set()

    props = h.properties
    if 'children' not in props:
        return np.array([], dtype=np.int64)

    parts = []
    for child in np.atleast_1d(props['children']):
        c = int(child)
        if c in _seen:
            continue
        _seen.add(c)
        try:
            ch = halo_cat[c]
        except Exception:
            continue
        parts.append(np.asarray(ch.dm['iord'], dtype=np.int64))
        grand = _child_dm_iords(ch, halo_cat, _seen)
        if len(grand):
            parts.append(grand)

    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def _clean_dm_iords(h, halo_cat, dm_iords=None):
    """DM iords of halo h with all child/sub-halo DM particles removed."""
    if dm_iords is None:
        dm_iords = np.asarray(h.dm['iord'], dtype=np.int64)
    child = _child_dm_iords(h, halo_cat)
    if len(child):
        dm_iords = dm_iords[~np.isin(dm_iords, child)]
    return dm_iords


def _has_stars(h):
    try:
        return len(h.st) > 0
    except Exception:
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simple AHF halo back-in-time tracker')
    parser.add_argument('sim_name', help='Simulation name (folder under the pynbody path)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='AHF (v1) halo number to seed on at z=0 (default: 1)')
    parser.add_argument('--prefer-stars', action='store_true',
                        help='Only consider halos that contain star particles as candidates')
    parser.add_argument('--window', type=int, default=None,
                        help='Restrict candidates to halonums within +/- this window of the '
                             'previous halo number (default: search all halos)')
    parser.add_argument('--dmo', action='store_true',
                        help='Use the DMO pynbody_path from config (default: hydro path)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_ahf_track.hdf5)')
    args = parser.parse_args()

    import h5py
    import pynbody
    import pynbody.halo.ahf

    if args.dmo:
        base_path = config.get_path('pynbody_path')
    else:
        base_path = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')
    pynbody_path = pjoin(base_path, args.sim_name)
    output_path = args.output or f'{args.sim_name}_ahf_track.hdf5'

    snap_paths = sorted(glob.glob(pjoin(pynbody_path, 'output_*')))
    if not snap_paths:
        snap_paths = sorted(glob.glob(pjoin(pynbody_path, '*')))
    outputs = [os.path.basename(p) for p in snap_paths]
    if not outputs:
        print(f'Error: no snapshots found in {pynbody_path}')
        sys.exit(1)

    print(f'Found {len(outputs)} snapshots in {pynbody_path}')
    print(f'Output: {output_path}')
    print(f'Seeding on AHF halo {args.halonumber} at z=0'
          + (', prefer-stars ON' if args.prefer_stars else ''))

    pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]

    outputs_reversed = outputs[::-1]  # z=0 first, then backward in time
    prev_iords = None
    prev_halonum = args.halonumber

    with h5py.File(output_path, 'w') as h5f:
        for output in outputs_reversed:
            print(f'\n-- {output} --')
            simfn = pjoin(pynbody_path, output)
            try:
                snap = pynbody.load(simfn)
                snap.physical_units()
            except Exception as e:
                print(f'  Failed to load snapshot: {e}, skipping')
                continue
            try:
                halo_cat = snap.halos(halo_numbers='v1')
            except Exception as e:
                print(f'  Failed to load AHF catalogue: {e}, skipping')
                del snap
                continue

            if prev_iords is None:
                # ── Seed at z=0 ──────────────────────────────────────────────
                try:
                    h = halo_cat[args.halonumber]
                except Exception as e:
                    print(f'  Could not load seed halo {args.halonumber}: {e}, aborting')
                    break
                iords = _clean_dm_iords(h, halo_cat)
                if len(iords) == 0:
                    print('  Seed halo has no DM particles after child removal, aborting')
                    break
                halonum = args.halonumber
                print(f'  Seeded halo {halonum}: {len(iords)} DM (child-removed)')
            else:
                # ── Find the best-overlap progenitor ─────────────────────────
                try:
                    n_halos = len(halo_cat)
                except Exception:
                    n_halos = 0
                if args.window is not None:
                    lo = max(1, prev_halonum - args.window)
                    hi = min(n_halos, prev_halonum + args.window)
                    candidates = range(lo, hi + 1)
                else:
                    candidates = range(1, n_halos + 1)

                best_num, best_iords, best_ov = None, None, 0
                for num in candidates:
                    try:
                        hc = halo_cat[num]
                    except Exception:
                        continue
                    if args.prefer_stars and not _has_stars(hc):
                        continue
                    hc_dm = np.asarray(hc.dm['iord'], dtype=np.int64)
                    # Raw overlap is an upper bound on the child-removed overlap
                    # (removing children can only lower it), so skip the (costly)
                    # child removal for halos that can't beat the current best.
                    raw_ov = int(np.isin(hc_dm, prev_iords).sum())
                    if raw_ov <= best_ov:
                        continue
                    clean = _clean_dm_iords(hc, halo_cat, dm_iords=hc_dm)
                    ov = int(np.isin(clean, prev_iords).sum())
                    if ov > best_ov:
                        best_ov, best_num, best_iords = ov, num, clean

                if best_num is None or best_ov == 0:
                    print('  No overlapping halo found, stopping tracking')
                    del snap
                    break
                halonum, iords = best_num, best_iords
                print(f'  Matched halo {halonum}: {len(iords)} DM (child-removed), '
                      f'overlap {best_ov} with previous')

            grp = h5f.require_group(output).require_group('main')
            grp.create_dataset('halonum', data=np.int64(halonum))
            grp.create_dataset('iords', data=iords.astype(np.int64))
            h5f.flush()

            prev_iords = iords
            prev_halonum = halonum
            del snap

    print(f'\nDone. Wrote {output_path}')


if __name__ == '__main__':
    main()
