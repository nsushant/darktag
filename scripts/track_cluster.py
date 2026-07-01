"""
Voxel-based merger tree tracker for DMO simulations.

Walks snapshots backwards in time (z=0 → high-z), loading each snapshot ONCE.
Tracks the main galaxy cluster and any significant satellite clusters via spatial
voxel clustering + iord overlap. New satellite branches are spawned dynamically
when a significant unclaimed cluster appears.

Uses iterative voxel degree expansion (scipy.ndimage.label) for fast, robust
cluster isolation without r200-dependent voxel sizing.

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
    python scripts/track_cluster.py Halo1459_HYDRO_Mreionx02 --ahf --voxel-size 0.5 --max-degree 20
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import _voxel_pick_cluster


# ─── Helpers ──────────────────────────────────────────────────────────────────

def voxel_all_clusters(positions, iords, voxel_size, degree=1):
    """
    Run ndimage.label at a fixed degree and return all clusters as a dict:
        root_label -> iords array

    Used for satellite discovery (we need ALL clusters, not just the best one).
    """
    from scipy.ndimage import label as ndimage_label

    if len(positions) == 0:
        return {}

    vx = np.floor(positions[:, 0] / voxel_size).astype(np.int64)
    vy = np.floor(positions[:, 1] / voxel_size).astype(np.int64)
    vz = np.floor(positions[:, 2] / voxel_size).astype(np.int64)

    ox, oy, oz = vx.min(), vy.min(), vz.min()
    gx, gy, gz = vx - ox, vy - oy, vz - oz

    nx, ny, nz = int(gx.max()) + 1, int(gy.max()) + 1, int(gz.max()) + 1
    grid = np.zeros((nx, ny, nz), dtype=bool)
    grid[gx, gy, gz] = True

    s = 2 * degree + 1
    labeled, n_clusters = ndimage_label(grid, structure=np.ones((s, s, s), dtype=bool))

    if n_clusters == 0:
        return {}

    particle_labels = labeled[gx, gy, gz]
    clusters = {}
    for lbl in range(1, n_clusters + 1):
        mask = particle_labels == lbl
        if mask.any():
            clusters[lbl] = iords[mask]

    return clusters


def majority_vote_halo(halo_cat, prev_halonum, window, prev_iords):
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
            h = halo_cat[idx]
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

def load_halo_catalogue(snap, use_ahf):
    """Load HOP or AHF halo catalogue from a pynbody snapshot."""
    import pynbody.halo.hop
    import pynbody.halo.ahf
    if use_ahf:
        pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
        return snap.halos(halo_numbers='v1')
    else:
        pynbody.config['halo-class-priority'] = [pynbody.halo.hop.HOPCatalogue]
        return pynbody.halo.hop.HOPCatalogue(snap)


def main():
    import argparse
    import h5py
    import pynbody
    import pynbody.halo.hop

    parser = argparse.ArgumentParser(description='Voxel merger tree tracker for DMO/HYDRO simulations')
    parser.add_argument('sim_name', help='Simulation name (e.g. Halo1459_DMO or Halo1459_HYDRO_Mreionx02)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo index at z=0 to seed main branch (default: 1)')
    parser.add_argument('--voxel-size', type=float, default=0.5,
                        help='Fixed voxel edge length in kpc (default: 0.5)')
    parser.add_argument('--max-degree', type=int, default=20,
                        help='Max voxel connectivity steps for iterative expansion (default: 20)')
    parser.add_argument('--size-jump', type=float, default=2.0,
                        help='Cluster size ratio that signals satellite absorption (default: 2.0)')
    parser.add_argument('--search-radius', type=float, default=3.0,
                        help='Load DM within this × r200_main of main centroid (default: 3.0)')
    parser.add_argument('--min-satellite-particles', type=int, default=100,
                        help='Min particle count for a new satellite branch (default: 100)')
    parser.add_argument('--halo-search-window', type=int, default=40,
                        help='±N window for HOP majority vote per branch (default: 40)')
    parser.add_argument('--sat-discovery-degree', type=int, default=1,
                        help='Fixed degree for satellite discovery clustering (default: 1)')
    parser.add_argument('--ahf', action='store_true',
                        help='Use AHF catalogue instead of HOP (required for HYDRO sims)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_cluster_tree.hdf5)')
    args = parser.parse_args()

    sim_name        = args.sim_name
    halonumber      = args.halonumber
    voxel_size      = args.voxel_size
    max_degree      = args.max_degree
    size_jump       = args.size_jump
    search_rad      = args.search_radius
    min_sat_p       = args.min_satellite_particles
    window          = args.halo_search_window
    sat_degree      = args.sat_discovery_degree
    use_ahf         = args.ahf
    output_path     = args.output or f'{sim_name}_cluster_tree.hdf5'

    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    snap_paths = sorted(glob.glob(pjoin(pynbody_path, 'output_*')))
    if len(snap_paths) == 0:
        snap_paths = sorted(glob.glob(pjoin(pynbody_path, '*')))
    outputs = [os.path.basename(p) for p in snap_paths]

    if len(outputs) == 0:
        print(f'Error: no snapshots found in {pynbody_path}')
        sys.exit(1)

    print(f'Found {len(outputs)} snapshots in {pynbody_path}')
    print(f'Output: {output_path}')
    print(f'Voxel size: {voxel_size} kpc, max_degree: {max_degree}, size_jump: {size_jump}')

    # ── Resume ────────────────────────────────────────────────────────────────
    active_branches = {}
    resume_from     = None

    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as f:
            done = set(f.keys())
            if done:
                outputs_reversed = outputs[::-1]
                for idx, out in enumerate(outputs_reversed):
                    if out in done:
                        last_done   = out
                        resume_from = idx + 1
                        grp = f[last_done]
                        for branch_id in grp.keys():
                            active_branches[branch_id] = {
                                'prev_iords':   grp[branch_id]['iords'][:],
                                'prev_halonum': int(grp[branch_id]['halonum'][()]),
                            }
                        break
        print(f'Resuming from after {last_done} — {len(active_branches)} active branches')

    # ── Main loop (z=0 first) ─────────────────────────────────────────────────
    outputs_reversed = outputs[::-1]
    start_idx        = resume_from or 0

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
                halo_cat = load_halo_catalogue(snap, use_ahf)
            except Exception as e:
                print(f'  Failed to load halo catalogue: {e}, skipping')
                continue

            # ── SEED at z=0 (first snapshot only) ────────────────────────────
            if not active_branches:
                if output != outputs_reversed[0]:
                    # Seeding failed at z=0 — don't retry on earlier snaps
                    print(f'  Seed not established at z=0, skipping')
                    del snap
                    continue

                try:
                    h = halo_cat[halonumber]
                except Exception as e:
                    print(f'  Could not load halo {halonumber}: {e}')
                    print(f'  Tip: for AHF use --halonumber 1 (1-indexed); for HOP try --halonumber 0')
                    del snap
                    break  # abort — no point trying earlier snaps without a seed

                pynbody.analysis.halo.center(h)
                r200 = get_r200(h)
                if r200 is None or r200 <= 0:
                    print(f'  Could not get r200, aborting')
                    del snap
                    break

                pos  = snap.dm['pos']
                dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
                dm_within = snap.dm[dist <= search_rad * r200]
                positions = np.array(dm_within['pos'])
                iords_all = np.array(dm_within['iord'])

                mask = _voxel_pick_cluster(
                    positions, iords_all, voxel_size,
                    prev_iords=None, max_degree=max_degree, size_jump=size_jump,
                )
                if mask is None or mask.sum() == 0:
                    print(f'  Seed clustering failed, skipping')
                    continue

                main_iords    = iords_all[mask]
                main_pos      = positions[mask]
                centroid      = main_pos.mean(axis=0)
                bounding_r    = float(np.sqrt(((main_pos - centroid)**2).sum(axis=1)).max())

                active_branches['main'] = {
                    'prev_iords':   main_iords,
                    'prev_halonum': halonumber,
                }

                grp = h5f.require_group(output)
                mg  = grp.require_group('main')
                mg.create_dataset('iords',          data=main_iords.astype(np.int64))
                mg.create_dataset('halonum',        data=np.int64(halonumber))
                mg.create_dataset('centroid',       data=centroid.astype(np.float64))
                mg.create_dataset('bounding_radius',data=np.float64(bounding_r))
                h5f.flush()
                print(f'  Seeded main branch: {len(main_iords)} particles '
                      f'(halo {halonumber}, r200={r200:.2f} kpc, '
                      f'bounding_r={bounding_r:.2f} kpc)')
                del snap, dm_within
                continue

            # ── SUBSEQUENT SNAPS ──────────────────────────────────────────────

            main_h, main_halonum = majority_vote_halo(
                halo_cat, active_branches['main']['prev_halonum'],
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

            pos   = snap.dm['pos']
            dist  = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
            dm_within  = snap.dm[dist <= search_rad * r200_main]
            positions  = np.array(dm_within['pos'])
            iords_all  = np.array(dm_within['iord'])

            if len(iords_all) == 0:
                print(f'  No DM particles within search radius, skipping')
                del snap, dm_within
                continue

            grp = h5f.require_group(output)
            claimed_iords = set()

            # Update all existing branches with iterative voxel refinement
            for branch_id, branch in list(active_branches.items()):
                h_b, halonum_b = majority_vote_halo(
                    halo_cat, branch['prev_halonum'], window, branch['prev_iords'])
                halonum_b = halonum_b if h_b is not None else branch['prev_halonum']

                mask = _voxel_pick_cluster(
                    positions, iords_all, voxel_size,
                    prev_iords=branch['prev_iords'],
                    max_degree=max_degree,
                    size_jump=size_jump,
                )

                if mask is None or mask.sum() == 0:
                    print(f'  Branch {branch_id}: no cluster found, dropping branch')
                    continue

                cluster_iords = iords_all[mask]
                cluster_pos   = positions[mask]
                centroid      = cluster_pos.mean(axis=0)
                bounding_r    = float(np.sqrt(((cluster_pos - centroid)**2).sum(axis=1)).max())
                claimed_iords.update(cluster_iords.tolist())

                branch['prev_iords']   = cluster_iords
                branch['prev_halonum'] = halonum_b

                bg = grp.require_group(branch_id)
                bg.create_dataset('iords',          data=cluster_iords.astype(np.int64))
                bg.create_dataset('halonum',        data=np.int64(halonum_b))
                bg.create_dataset('centroid',       data=centroid.astype(np.float64))
                bg.create_dataset('bounding_radius',data=np.float64(bounding_r))

                r200_b_str = f'{get_r200(h_b):.2f}' if h_b is not None else 'n/a'
                print(f'  Branch {branch_id}: {len(cluster_iords)} particles '
                      f'(halo {halonum_b}, r200={r200_b_str} kpc, '
                      f'bounding_r={bounding_r:.2f} kpc)')

            # Discover new satellite branches from unclaimed clusters
            all_clusters = voxel_all_clusters(
                positions, iords_all, voxel_size, degree=sat_degree)

            n_sat = sum(1 for k in active_branches if k != 'main')
            for lbl, sat_iords in all_clusters.items():
                if len(sat_iords) < min_sat_p:
                    continue
                # skip if mostly claimed by existing branches
                overlap_claimed = np.isin(sat_iords, list(claimed_iords)).sum()
                if overlap_claimed > 0.5 * len(sat_iords):
                    continue

                sat_id      = f'sat_{n_sat:03d}'
                sat_pos     = positions[np.isin(iords_all, sat_iords)]
                sat_centroid   = sat_pos.mean(axis=0)
                sat_bounding_r = float(np.sqrt(((sat_pos - sat_centroid)**2).sum(axis=1)).max())
                h_sat, halonum_sat = majority_vote_halo(halo_cat, 0, window, sat_iords)

                active_branches[sat_id] = {
                    'prev_iords':   sat_iords,
                    'prev_halonum': halonum_sat,
                }
                claimed_iords.update(sat_iords.tolist())
                n_sat += 1

                sg = grp.require_group(sat_id)
                sg.create_dataset('iords',          data=sat_iords.astype(np.int64))
                sg.create_dataset('halonum',        data=np.int64(halonum_sat))
                sg.create_dataset('centroid',       data=sat_centroid.astype(np.float64))
                sg.create_dataset('bounding_radius',data=np.float64(sat_bounding_r))
                print(f'  Spawned new branch {sat_id}: {len(sat_iords)} particles '
                      f'(halo {halonum_sat}, bounding_r={sat_bounding_r:.2f} kpc)')

            h5f.flush()
            del snap, dm_within

    finally:
        h5f.close()

    print(f'\nDone. Written to {output_path}')
    print(f'Total branches: {len(active_branches)}')


if __name__ == '__main__':
    main()
