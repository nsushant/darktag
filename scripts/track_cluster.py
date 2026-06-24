"""
Voxel-based cluster tracker for DMO simulations.

Walks snapshots backwards in time (z=0 → high-z), tracking the main galaxy's
DM cluster via spatial voxel clustering + iord overlap. Writes one HDF5 file
with a dataset per snapshot containing the iords of the tracked cluster.

The output can be consumed by the tagging scripts and hydro reff scripts to
restrict particle selection to the identified cluster at each snapshot.

Usage:
    python scripts/track_cluster.py Halo1459_DMO
    python scripts/track_cluster.py Halo1459_DMO --halonumber 0 --voxel-fraction 0.20
"""

import sys
import os
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


# ─── Voxel clustering ─────────────────────────────────────────────────────────

def voxel_cluster(x, y, z, voxel_size):
    """
    Assign particles to connected voxel clusters.

    Returns
    -------
    particle_labels : int32 array, length n_particles
        Cluster root ID for each particle (-1 if isolated, but union-find
        means every occupied voxel is in some cluster).
    clusters : dict  root_id -> {"count": int, "indices": int array}
        particle indices belonging to each cluster.
    """
    ix = np.floor(x / voxel_size).astype(np.int32)
    iy = np.floor(y / voxel_size).astype(np.int32)
    iz = np.floor(z / voxel_size).astype(np.int32)

    ix -= ix.min()
    iy -= iy.min()
    iz -= iz.min()

    key = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

    order       = np.argsort(key)
    key_sorted  = key[order]

    change  = np.diff(key_sorted) != 0
    starts  = np.r_[0, np.where(change)[0] + 1]
    ends    = np.r_[starts[1:], len(key_sorted)]

    # voxel grid coords
    vx = ix[order][starts]
    vy = iy[order][starts]
    vz = iz[order][starts]
    vkey = vx * 73856093 ^ vy * 19349663 ^ vz * 83492791
    lookup = {k: i for i, k in enumerate(vkey)}

    # union-find
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

    # map particles → voxel index
    particle_voxel = np.empty(len(x), dtype=np.int32)
    for v, (s, e) in enumerate(zip(starts, ends)):
        particle_voxel[order[s:e]] = v

    # build clusters
    clusters = {}
    for i in range(len(vkey)):
        r = find(i)
        idxs = np.where(particle_voxel == i)[0]
        if r not in clusters:
            clusters[r] = {"count": 0, "indices": np.array([], dtype=np.int64)}
        clusters[r]["count"]   += len(idxs)
        clusters[r]["indices"]  = np.concatenate([clusters[r]["indices"], idxs])

    # assign particle label = cluster root
    particle_labels = np.empty(len(x), dtype=np.int32)
    for i in range(len(vkey)):
        r = find(i)
        idxs = np.where(particle_voxel == i)[0]
        particle_labels[idxs] = r

    return particle_labels, clusters


def pick_cluster_by_overlap(all_iords, clusters, prev_iords):
    """
    Among all clusters, return the indices of particles in the cluster
    with the most iord overlap with prev_iords.
    Falls back to largest cluster by count if no overlap found.
    """
    prev_set = set(prev_iords)
    best_root    = None
    best_overlap = -1
    best_count   = -1

    for root, info in clusters.items():
        cluster_iords = all_iords[info["indices"]]
        overlap = np.isin(cluster_iords, list(prev_set)).sum()
        if overlap > best_overlap or (overlap == best_overlap and info["count"] > best_count):
            best_overlap = overlap
            best_count   = info["count"]
            best_root    = root

    if best_root is None:
        return np.array([], dtype=np.int64)

    return all_iords[clusters[best_root]["indices"]]


def majority_vote_halo(hop_cat, prev_halonum, window, prev_iords):
    """
    Search halos in [prev_halonum-window, prev_halonum+window].
    Return (halo_obj, halo_index) with the most iord overlap with prev_iords.
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


def get_r200(h, snap):
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
    import tangos

    parser = argparse.ArgumentParser(description='Voxel cluster tracker for DMO simulations')
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--halonumber', type=int, default=0,
                        help='HOP halo index at z=0 (0-based, default: 0)')
    parser.add_argument('--voxel-fraction', type=float, default=0.20,
                        help='Voxel size as fraction of r200 (default: 0.20)')
    parser.add_argument('--halo-search-window', type=int, default=40,
                        help='±N halos to search around prev halo number (default: 40)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_cluster_iords.hdf5)')
    args = parser.parse_args()

    sim_name     = args.sim_name
    halonumber   = args.halonumber
    voxel_frac   = args.voxel_fraction
    window       = args.halo_search_window
    output_path  = args.output or f'{sim_name}_cluster_iords.hdf5'

    tangos_path  = config.get_path('tangos_path')
    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    # get snapshot list via progenitor chain
    main_halo   = sim.timesteps[-1].halos[halonumber]
    t_all       = main_halo.calculate_for_progenitors('t()')[0][::-1]
    outputs_all = np.array([ts.__dict__['extension'] for ts in sim.timesteps])
    times_all   = np.array([ts.__dict__['time_gyr']  for ts in sim.timesteps])
    outputs     = outputs_all[np.isin(times_all, t_all)]
    outputs.sort()

    print(f'Tracking {len(outputs)} snapshots for {sim_name}')
    print(f'Output: {output_path}')

    # load existing datasets for resume
    existing = set()
    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as f:
            existing = set(f.keys())
        print(f'Resuming — {len(existing)} snapshots already done')

    prev_iords   = None
    prev_halonum = halonumber

    # open HDF5 in append mode
    h5f = h5py.File(output_path, 'a')

    try:
        # iterate backwards: outputs[-1] = z=0, outputs[0] = highest z
        for i, output in enumerate(outputs[::-1]):

            if output in existing:
                print(f'  {output} already done, skipping')
                # still need to seed prev_iords from file for continuity
                if prev_iords is None:
                    prev_iords = h5f[output][:]
                continue

            print(f'  Processing {output} ...')

            simfn = pjoin(pynbody_path, output)
            try:
                snap = pynbody.load(simfn)
            except Exception as e:
                print(f'    Failed to load snapshot: {e}, skipping')
                continue

            snap.physical_units()

            try:
                hop_cat = pynbody.halo.hop.HOPCatalogue(snap)
            except Exception as e:
                print(f'    Failed to load HOP catalogue: {e}, skipping')
                continue

            # ── z=0: seed from specified halo ────────────────────────────────
            if prev_iords is None:
                try:
                    h = hop_cat[halonumber]
                    current_halonum = halonumber
                except Exception as e:
                    print(f'    Could not load halo {halonumber}: {e}, skipping')
                    continue
            # ── earlier snaps: majority vote ─────────────────────────────────
            else:
                h, current_halonum = majority_vote_halo(hop_cat, prev_halonum, window, prev_iords)
                if h is None:
                    print(f'    No matching halo found, skipping')
                    continue

            pynbody.analysis.halo.center(h)

            r200 = get_r200(h, snap)
            if r200 is None or r200 <= 0:
                print(f'    Could not determine r200, skipping')
                continue

            # select all DM within r200
            pos  = snap.dm['pos']
            dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
            mask = dist <= r200
            dm_within = snap.dm[mask]

            if len(dm_within) == 0:
                print(f'    No DM particles within r200, skipping')
                continue

            x = np.array(dm_within['x'])
            y = np.array(dm_within['y'])
            z = np.array(dm_within['z'])
            iords = np.array(dm_within['iord'])

            voxel_size = voxel_frac * r200
            _, clusters = voxel_cluster(x, y, z, voxel_size)

            if prev_iords is None:
                # z=0: pick largest cluster
                best_root    = max(clusters, key=lambda r: clusters[r]["count"])
                cluster_iords = iords[clusters[best_root]["indices"]]
            else:
                cluster_iords = pick_cluster_by_overlap(iords, clusters, prev_iords)

            if len(cluster_iords) == 0:
                print(f'    Empty cluster result, skipping')
                continue

            print(f'    Cluster size: {len(cluster_iords)} particles  (halo {current_halonum}, r200={r200:.2f})')

            h5f.create_dataset(output, data=cluster_iords.astype(np.int64))
            h5f.flush()

            prev_iords   = cluster_iords
            prev_halonum = current_halonum

    finally:
        h5f.close()

    print(f'\nDone. Written to {output_path}')


if __name__ == '__main__':
    main()
