"""
Voxel-based merger tree tracker for DMO simulations.

Walks snapshots backwards in time (z=0 → high-z), loading each snapshot ONCE.
Tracks the main galaxy cluster and any significant satellite clusters via spatial
voxel clustering + iord overlap. New satellite branches are spawned dynamically
when a significant unclaimed cluster appears.

Output HDF5 structure:
    /output_0089/main/iords      → int64 array
    /output_0089/main/halonum    → int scalar
    /output_0088/main/iords
    /output_0088/main/halonum
    /output_0088/sat_000/iords   ← spawned when first seen going backwards
    /output_0088/sat_000/halonum
    ...

Usage:
    python scripts/track_cluster.py Halo1459_DMO
    python scripts/track_cluster.py Halo1459_DMO --voxel-fraction 0.20 --search-radius 3.0
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


# ─── Helpers ──────────────────────────────────────────────────────────────────

def voxel_cluster(x, y, z, voxel_size):
    """
    Assign particles to connected voxel clusters via union-find on 26-neighbours.

    Returns
    -------
    clusters : dict  root_id -> {"count": int, "indices": int64 array}
    """
    ix = np.floor(x / voxel_size).astype(np.int32)
    iy = np.floor(y / voxel_size).astype(np.int32)
    iz = np.floor(z / voxel_size).astype(np.int32)

    ix -= ix.min()
    iy -= iy.min()
    iz -= iz.min()

    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

    order      = np.argsort(key)
    key_sorted = key[order]

    change = np.diff(key_sorted) != 0
    starts = np.r_[0, np.where(change)[0] + 1]
    ends   = np.r_[starts[1:], len(key_sorted)]

    vx   = ix[order][starts]
    vy   = iy[order][starts]
    vz   = iz[order][starts]
    vkey = vx * 73856093 ^ vy * 19349663 ^ vz * 83492791
    lookup = {k: i for i, k in enumerate(vkey)}

    parent = np.arange(len(vkey), dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (x0, y0, z0) in enumerate(zip(vx, vy, vz)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nk = (x0+dx)*73856093 ^ (y0+dy)*19349663 ^ (z0+dz)*83492791
                    j = lookup.get(nk)
                    if j is not None:
                        union(i, j)

    particle_voxel = np.empty(len(x), dtype=np.int32)
    for v, (s, e) in enumerate(zip(starts, ends)):
        particle_voxel[order[s:e]] = v

    clusters = {}
    for i in range(len(vkey)):
        r = find(i)
        idxs = np.where(particle_voxel == i)[0]
        if r not in clusters:
            clusters[r] = {"count": 0, "indices": np.array([], dtype=np.int64)}
        clusters[r]["count"]   += len(idxs)
        clusters[r]["indices"]  = np.concatenate([clusters[r]["indices"], idxs])

    return clusters


def pick_cluster_by_overlap(all_iords, clusters, prev_iords):
    """
    Return (root, iords) of the cluster with most iord overlap with prev_iords.
    Falls back to largest cluster by count if no overlap.
    """
    prev_set     = set(prev_iords.tolist())
    best_root    = None
    best_overlap = -1
    best_count   = -1

    for root, info in clusters.items():
        cluster_iords = all_iords[info["indices"]]
        overlap = int(np.isin(cluster_iords, list(prev_set)).sum())
        if overlap > best_overlap or (overlap == best_overlap and info["count"] > best_count):
            best_overlap = overlap
            best_count   = info["count"]
            best_root    = root

    if best_root is None:
        return None, np.array([], dtype=np.int64)

    return best_root, all_iords[clusters[best_root]["indices"]]


def majority_vote_halo(hop_cat, prev_halonum, window, prev_iords):
    """
    Search halos in [prev_halonum-window, prev_halonum+window].
    Return (halo_obj, halo_index) with most iord overlap with prev_iords.
    """
    lo = max(0, prev_halonum - window)
    hi = prev_halonum + window

    best_halo  = None
    best_idx   = prev_halonum
    best_score = -1

    for idx in range(lo, hi + 1):
        try:
            h = hop_cat[idx]
            score = len(np.intersect1d(h.dm['iord'], prev_iords))
            if score > best_score:
                best_score = score
                best_halo  = h
                best_idx   = idx
        except Exception:
            continue

    return best_halo, best_idx


def get_r200(h):
    """Read r200 from halo properties, fallback to pynbody virial radius calc."""
    import pynbody
    for key in ('Rvir', 'r200', 'Rhalo'):
        val = h.properties.get(key)
        if val is not None and float(val) > 0:
            return float(val)
    try:
        return float(pynbody.analysis.halo.virial_radius(h))
    except Exception:
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import h5py
    import pynbody
    import pynbody.halo.hop

    parser = argparse.ArgumentParser(description='Voxel merger tree tracker for DMO simulations')
    parser.add_argument('sim_name', help='Simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--halonumber', type=int, default=0,
                        help='HOP halo index at z=0 to seed main branch (default: 0)')
    parser.add_argument('--voxel-fraction', type=float, default=0.20,
                        help='Voxel size as fraction of r200_main (default: 0.20)')
    parser.add_argument('--search-radius', type=float, default=3.0,
                        help='Load DM within this × r200_main of main centroid (default: 3.0)')
    parser.add_argument('--min-satellite-particles', type=int, default=100,
                        help='Min particle count for a new satellite branch (default: 100)')
    parser.add_argument('--halo-search-window', type=int, default=40,
                        help='±N window for HOP majority vote per branch (default: 40)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_cluster_tree.hdf5)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    halonumber  = args.halonumber
    voxel_frac  = args.voxel_fraction
    search_rad  = args.search_radius
    min_sat_p   = args.min_satellite_particles
    window      = args.halo_search_window
    output_path = args.output or f'{sim_name}_cluster_tree.hdf5'

    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    # Discover snapshots by globbing the pynbody directory
    snap_paths = sorted(glob.glob(pjoin(pynbody_path, 'output_*')))
    if len(snap_paths) == 0:
        # fallback: look for numbered directories
        snap_paths = sorted(glob.glob(pjoin(pynbody_path, '*')))
    outputs = [os.path.basename(p) for p in snap_paths]

    if len(outputs) == 0:
        print(f'Error: no snapshots found in {pynbody_path}')
        sys.exit(1)

    print(f'Found {len(outputs)} snapshots in {pynbody_path}')
    print(f'Output: {output_path}')

    # ── Resume: reconstruct active_branches from last written snapshot ─────────
    active_branches = {}   # branch_id -> {prev_iords, prev_halonum}
    resume_from    = None  # index in outputs[::-1] to resume from

    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as f:
            done = set(f.keys())
            if done:
                # find the last processed snapshot (most recent in backwards order)
                outputs_reversed = outputs[::-1]
                for idx, out in enumerate(outputs_reversed):
                    if out in done:
                        last_done = out
                        resume_from = idx + 1  # start from next one
                        grp = f[last_done]
                        for branch_id in grp.keys():
                            active_branches[branch_id] = {
                                'prev_iords':   grp[branch_id]['iords'][:],
                                'prev_halonum': int(grp[branch_id]['halonum'][()]),
                            }
                        break
        print(f'Resuming from after {last_done} — {len(active_branches)} active branches')

    # ── Main loop (z=0 first) ──────────────────────────────────────────────────
    outputs_reversed = outputs[::-1]
    start_idx = resume_from or 0

    h5f = h5py.File(output_path, 'a')

    try:
        for output in outputs_reversed[start_idx:]:
            print(f'\n── {output} ──')

            simfn = pjoin(pynbody_path, output)
            try:
                snap = pynbody.load(simfn)
                snap.physical_units()
            except Exception as e:
                print(f'  Failed to load snapshot: {e}, skipping')
                continue

            try:
                hop_cat = pynbody.halo.hop.HOPCatalogue(snap)
            except Exception as e:
                print(f'  Failed to load HOP catalogue: {e}, skipping')
                continue

            # ── SEED at z=0 ───────────────────────────────────────────────────
            if not active_branches:
                try:
                    h = hop_cat[halonumber]
                except Exception as e:
                    print(f'  Could not load halo {halonumber}: {e}, skipping')
                    continue

                pynbody.analysis.halo.center(h)
                r200 = get_r200(h)
                if r200 is None or r200 <= 0:
                    print(f'  Could not get r200, skipping')
                    continue

                pos  = snap.dm['pos']
                dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
                dm_within  = snap.dm[dist <= search_rad * r200]
                iords_all  = np.array(dm_within['iord'])

                x = np.array(dm_within['x'])
                y = np.array(dm_within['y'])
                z_arr = np.array(dm_within['z'])
                clusters = voxel_cluster(x, y, z_arr, voxel_frac * r200)

                best_root = max(clusters, key=lambda r: clusters[r]["count"])
                main_iords = iords_all[clusters[best_root]["indices"].astype(int)]

                active_branches['main'] = {
                    'prev_iords':   main_iords,
                    'prev_halonum': halonumber,
                }

                grp = h5f.require_group(output)
                mg  = grp.require_group('main')
                mg.create_dataset('iords',   data=main_iords.astype(np.int64))
                mg.create_dataset('halonum', data=np.int64(halonumber))
                h5f.flush()
                print(f'  Seeded main branch: {len(main_iords)} particles (halo {halonumber}, r200={r200:.2f})')
                del snap, dm_within
                continue

            # ── SUBSEQUENT SNAPS ──────────────────────────────────────────────

            # Main branch majority vote → centre + r200
            main_h, main_halonum = majority_vote_halo(
                hop_cat, active_branches['main']['prev_halonum'],
                window,  active_branches['main']['prev_iords'])

            if main_h is None:
                print(f'  Main branch: no matching halo found, skipping snap')
                del snap
                continue

            pynbody.analysis.halo.center(main_h)
            r200_main = get_r200(main_h)
            if r200_main is None or r200_main <= 0:
                print(f'  Could not get r200_main, skipping snap')
                del snap
                continue

            # Load all DM within search_radius * r200_main (single load)
            pos  = snap.dm['pos']
            dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
            dm_within = snap.dm[dist <= search_rad * r200_main]
            iords_all = np.array(dm_within['iord'])

            if len(iords_all) == 0:
                print(f'  No DM particles within search radius, skipping')
                del snap, dm_within
                continue

            x     = np.array(dm_within['x'])
            y     = np.array(dm_within['y'])
            z_arr = np.array(dm_within['z'])

            # Shared clustering at r200_main voxel size — used for satellite discovery
            clusters_main = voxel_cluster(x, y, z_arr, voxel_frac * r200_main)

            # Update all existing branches
            claimed_roots = set()
            grp = h5f.require_group(output)

            for branch_id, branch in list(active_branches.items()):
                # majority vote first → gives us h_b and its r200
                h_b, halonum_b = majority_vote_halo(
                    hop_cat, branch['prev_halonum'], window, branch['prev_iords'])

                if h_b is not None:
                    r200_b = get_r200(h_b)
                else:
                    r200_b = None
                    halonum_b = branch['prev_halonum']

                # per-branch clustering with this branch's own r200
                if r200_b is not None and r200_b > 0:
                    clusters_b = voxel_cluster(x, y, z_arr, voxel_frac * r200_b)
                else:
                    clusters_b = clusters_main  # fallback

                best_root, cluster_iords = pick_cluster_by_overlap(
                    iords_all, clusters_b, branch['prev_iords'])

                if best_root is None or len(cluster_iords) == 0:
                    print(f'  Branch {branch_id}: no cluster found, dropping branch')
                    continue

                # mark root in main clustering as claimed to suppress satellite re-discovery
                _, main_root_iords = pick_cluster_by_overlap(
                    iords_all, clusters_main, cluster_iords)
                # find which main-cluster root overlaps most with these iords
                for root, info in clusters_main.items():
                    if len(np.intersect1d(iords_all[info['indices'].astype(int)], cluster_iords)) > 0:
                        claimed_roots.add(root)

                branch['prev_iords']   = cluster_iords
                branch['prev_halonum'] = halonum_b

                bg = grp.require_group(branch_id)
                bg.create_dataset('iords',   data=cluster_iords.astype(np.int64))
                bg.create_dataset('halonum', data=np.int64(halonum_b))

                r200_b_str = f'{r200_b:.2f}' if r200_b else 'n/a'
                print(f'  Branch {branch_id}: {len(cluster_iords)} particles (halo {halonum_b}, r200={r200_b_str})')

            # Discover new satellite branches from unclaimed clusters (using main clustering)
            n_sat = len(active_branches) - 1  # -1 for 'main'
            for root, info in clusters_main.items():
                if root in claimed_roots:
                    continue
                if info['count'] < min_sat_p:
                    continue

                sat_id    = f'sat_{n_sat:03d}'
                sat_iords = iords_all[info["indices"].astype(int)]
                h_sat, halonum_sat = majority_vote_halo(hop_cat, 0, window, sat_iords)

                active_branches[sat_id] = {
                    'prev_iords':   sat_iords,
                    'prev_halonum': halonum_sat,
                }
                claimed_roots.add(root)
                n_sat += 1

                sg = grp.require_group(sat_id)
                sg.create_dataset('iords',   data=sat_iords.astype(np.int64))
                sg.create_dataset('halonum', data=np.int64(halonum_sat))

                print(f'  Spawned new branch {sat_id}: {len(sat_iords)} particles (halo {halonum_sat})')

            h5f.flush()
            del snap, dm_within

    finally:
        h5f.close()

    print(f'\nDone. Written to {output_path}')
    print(f'Total branches: {len(active_branches)}')


if __name__ == '__main__':
    main()
