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
    python scripts/track_cluster.py Halo1459_HYDRO_Mreionx02 --ahf --voxel-size 0.2
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config
from darktag.tagging.tagging_wrapper_func import _density_region_grow


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clip_to_cluster(positions, iords, prev_iords, padding_factor=2.0, min_radius_kpc=5.0):
    """
    Restrict positions/iords to a sphere around where the previous cluster iords
    currently are, with a padding_factor * bounding_radius buffer.

    This prevents the voxel grid from spanning the full search sphere (which can be
    many GiB for large halos at fine voxel resolution).

    Falls back to the full arrays if prev_iords cannot be located in the current snap.
    """
    if prev_iords is None or len(prev_iords) == 0:
        return positions, iords

    in_prev = np.isin(iords, prev_iords)
    if not in_prev.any():
        return positions, iords

    prev_pos   = positions[in_prev]
    centroid   = prev_pos.mean(axis=0)
    bounding_r = float(np.sqrt(((prev_pos - centroid) ** 2).sum(axis=1)).max())
    clip_r     = max(bounding_r * padding_factor, min_radius_kpc)

    dist = np.sqrt(((positions - centroid) ** 2).sum(axis=1))
    mask = dist <= clip_r
    return positions[mask], iords[mask]


def voxel_all_clusters(positions, iords, voxel_size, degree=1, max_grid_gb=8.0):
    """
    Run ndimage.label at a fixed degree and return all clusters as a dict:
        root_label -> iords array

    Used for satellite discovery (we need ALL clusters, not just the best one).
    """
    from scipy.ndimage import label as ndimage_label, binary_dilation

    if len(positions) == 0:
        return {}

    vx = np.floor(positions[:, 0] / voxel_size).astype(np.int64)
    vy = np.floor(positions[:, 1] / voxel_size).astype(np.int64)
    vz = np.floor(positions[:, 2] / voxel_size).astype(np.int64)

    ox, oy, oz = int(vx.min()), int(vy.min()), int(vz.min())
    gx = (vx - ox).astype(np.int32)
    gy = (vy - oy).astype(np.int32)
    gz = (vz - oz).astype(np.int32)
    del vx, vy, vz

    nx, ny, nz = int(gx.max()) + 1, int(gy.max()) + 1, int(gz.max()) + 1
    grid_gb = nx * ny * nz / 1e9  # bool = 1 byte
    if grid_gb > max_grid_gb:
        print(f'  voxel_all_clusters: grid ({nx}×{ny}×{nz}) would require '
              f'{grid_gb:.1f} GB, skipping satellite discovery for this snap')
        return {}

    grid = np.zeros((nx, ny, nz), dtype=bool)
    grid[gx, gy, gz] = True

    structure3 = np.ones((3, 3, 3), dtype=bool)
    expanded = binary_dilation(grid, structure=structure3, iterations=degree) if degree > 0 else grid
    del grid
    labeled, n_clusters = ndimage_label(expanded, structure=structure3)
    del expanded

    if n_clusters == 0:
        return {}

    # Extract per-particle labels, free the large labeled grid immediately
    particle_labels = labeled[gx, gy, gz]
    del labeled

    # Build cluster dict without a Python loop — sort by label then split
    sort_idx      = np.argsort(particle_labels, kind='stable')
    sorted_labels = particle_labels[sort_idx]
    sorted_iords  = iords[sort_idx]
    unique_lbls, counts = np.unique(sorted_labels, return_counts=True)
    splits = np.split(sorted_iords, np.cumsum(counts)[:-1])
    clusters = {int(lbl): arr for lbl, arr in zip(unique_lbls, splits) if lbl > 0}

    return clusters


def majority_vote_halo(halo_cat, prev_halonum, window, prev_iords, prefer_stars=False):
    """
    Search halos in [prev_halonum-window, prev_halonum+window].
    Return (halo_obj, halo_index) with most iord overlap with prev_iords.

    If prefer_stars=True, restrict candidates to halos that contain at least
    one stellar particle before picking the best overlap. If no candidate has
    stars (e.g. before first star formation), falls back to the normal winner.
    """
    lo = max(0, prev_halonum - window)
    hi = prev_halonum + window

    candidates = []  # list of (score, idx, halo_obj, has_stars)

    for idx in range(lo, hi + 1):
        try:
            h = halo_cat[idx]
            score = len(np.intersect1d(h.dm['iord'], prev_iords))
            if score == 0:
                continue
            has_stars = False
            if prefer_stars:
                try:
                    has_stars = len(h.st) > 0
                except Exception:
                    pass
            candidates.append((score, idx, h, has_stars))
        except Exception:
            continue

    if not candidates:
        return None, prev_halonum

    if prefer_stars:
        starred = [(s, i, h) for s, i, h, hs in candidates if hs]
        pool = starred if starred else [(s, i, h) for s, i, h, _ in candidates]
        if not starred:
            print('  majority_vote_halo: no candidate has stars, using best overlap')
    else:
        pool = [(s, i, h) for s, i, h, _ in candidates]

    best = max(pool, key=lambda x: x[0])
    return best[2], best[1]


def centroid_fallback_halo(halo_cat, prev_centroid, prev_iords, max_centroid_dist=500.0):
    """
    Fallback halo finder using spatial proximity of AHF halo centres.
    Reads halo centres from AHF properties (no particle loading).
    Among halos within max_centroid_dist kpc of prev_centroid, returns
    the one with most iord overlap with prev_iords.
    """
    best_halo  = None
    best_idx   = None
    best_score = 0

    for idx in range(1, len(halo_cat) + 1):
        try:
            h = halo_cat[idx]
            props = h.properties
            xc = float(props.get('Xc', props.get('Xcmbp', None)))
            yc = float(props.get('Yc', props.get('Ycmbp', None)))
            zc = float(props.get('Zc', props.get('Zcmbp', None)))
            dist = np.sqrt((xc - prev_centroid[0])**2 +
                           (yc - prev_centroid[1])**2 +
                           (zc - prev_centroid[2])**2)
            if dist > max_centroid_dist:
                continue
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


def find_hop_halonum(snap, cluster_iords, hop_cat=None, max_halos=50):
    """
    Find the HOP halo with the most iord overlap with cluster_iords.
    Returns the 1-based HOP halo number (for tangos compatibility), or None.

    If hop_cat is provided it is reused (the caller is responsible for building
    it once per snapshot); otherwise a HOP catalogue is constructed here.
    """
    import pynbody.halo.hop
    if hop_cat is None:
        try:
            hop_cat = pynbody.halo.hop.HOPCatalogue(snap)
        except Exception as e:
            print(f'  Could not load HOP catalogue: {e}')
            return None

    # Build the comparison array ONCE (sorted unique) instead of rebuilding a
    # Python list on every loop iteration.
    cluster_arr = np.unique(np.asarray(cluster_iords).ravel())
    n_cluster = len(cluster_arr)
    best_idx   = None
    best_score = 0

    for idx in range(min(max_halos, len(hop_cat))):
        try:
            h = hop_cat[idx]
            overlap = np.isin(np.asarray(h.dm['iord']), cluster_arr).sum()
            if overlap > best_score:
                best_score = overlap
                best_idx   = idx
                if best_score > 0.5 * n_cluster:
                    break
        except Exception:
            continue

    if best_idx is not None:
        # HOP catalogue is 0-indexed, tangos uses 1-based halo numbers
        hop_halonum = best_idx + 1
        print(f'  HOP halo match: halo {hop_halonum} (overlap {best_score}/{n_cluster})')
        return hop_halonum
    return None


def main():
    import argparse
    import h5py
    import pynbody
    import pynbody.halo.hop

    parser = argparse.ArgumentParser(description='Voxel merger tree tracker for DMO/HYDRO simulations')
    parser.add_argument('sim_name', help='Simulation name (e.g. Halo1459_DMO or Halo1459_HYDRO_Mreionx02)')
    parser.add_argument('--halonumber', type=int, default=1,
                        help='Halo index at z=0 to seed main branch (default: 1)')
    parser.add_argument('--voxel-size', type=float, default=0.2,
                        help='Fixed voxel edge length in kpc (default: 0.2)')
    parser.add_argument('--min-cluster-size', type=int, default=20,
                        help='HDBSCAN min_cluster_size in voxels (default: 20)')
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
    parser.add_argument('--dmo', action='store_true',
                        help='Use DMO pynbody_path from config (default: use hydro_pynbody_path)')
    parser.add_argument('--centroid', action='store_true',
                        help='Fall back to centroid-based halo search if majority vote fails')
    parser.add_argument('--centroid-max-dist', type=float, default=500.0,
                        help='Max distance (kpc) from previous centroid for fallback search (default: 500)')
    parser.add_argument('--wall-time', type=float, default=None,
                        help='Stop N minutes before wall time limit to exit gracefully (default: no limit)')
    parser.add_argument('--prefer-stars', action='store_true',
                        help='Prefer halos with stellar particles when multiple candidates have DM overlap '
                             '(useful for HYDRO sims to avoid tracking star-free DM halos)')
    parser.add_argument('--no-satellites', action='store_true',
                        help='Only track the main branch, skip satellite discovery (much faster)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_cluster_tree.hdf5)')
    args = parser.parse_args()

    sim_name           = args.sim_name
    halonumber         = args.halonumber
    voxel_size         = args.voxel_size
    min_cluster_size   = args.min_cluster_size
    search_rad         = args.search_radius
    min_sat_p       = args.min_satellite_particles
    window          = args.halo_search_window
    sat_degree      = args.sat_discovery_degree
    use_ahf         = args.ahf
    use_centroid      = args.centroid
    centroid_max_dist = args.centroid_max_dist
    wall_time         = args.wall_time
    prefer_stars      = args.prefer_stars
    no_satellites     = args.no_satellites
    output_path       = args.output or f'{sim_name}_cluster_tree.hdf5'

    if args.dmo:
        _base_path = config.get_path('pynbody_path')
    else:
        _base_path = config.get_with_default('paths', 'hydro_pynbody_path', None) or config.get_path('pynbody_path')
    pynbody_path = pjoin(_base_path, sim_name)

    snap_paths = sorted(glob.glob(pjoin(pynbody_path, 'output_*')))
    if len(snap_paths) == 0:
        snap_paths = sorted(glob.glob(pjoin(pynbody_path, '*')))
    outputs = [os.path.basename(p) for p in snap_paths]

    if len(outputs) == 0:
        print(f'Error: no snapshots found in {pynbody_path}')
        sys.exit(1)

    print(f'Found {len(outputs)} snapshots in {pynbody_path}')
    print(f'Output: {output_path}')
    print(f'Voxel size: {voxel_size} kpc, min_cluster_size: {min_cluster_size}')

    # ── Resume ────────────────────────────────────────────────────────────────
    active_branches = {}
    resume_from     = None

    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as f:
            done = set(f.keys())
            if done:
                outputs_reversed = outputs[::-1]
                # Find the earliest VALID processed snapshot (skip partial/empty groups)
                last_done   = None
                resume_from = None
                for idx, out in enumerate(outputs_reversed):
                    if out in done:
                        grp_check = f[out]
                        # Only accept if main branch has iords (i.e. was fully written)
                        if 'main' in grp_check and 'iords' in grp_check['main']:
                            last_done   = out
                            resume_from = idx + 1
                        # keep scanning — we want the last valid match (earliest time)
                if last_done is not None:
                    grp = f[last_done]
                    for branch_id in grp.keys():
                        active_branches[branch_id] = {
                            'prev_iords':    grp[branch_id]['iords'][:],
                            'prev_halonum':  int(grp[branch_id]['halonum'][()]),
                            'prev_centroid': grp[branch_id]['centroid'][:] if 'centroid' in grp[branch_id] else None,
                        }
        print(f'Resuming from after {last_done} — {len(active_branches)} active branches')

    # ── Main loop (z=0 first) ─────────────────────────────────────────────────
    outputs_reversed = outputs[::-1]
    start_idx        = resume_from or 0

    h5f = h5py.File(output_path, 'a')
    t_start = __import__('time').time()

    try:
        for output in outputs_reversed[start_idx:]:
            if wall_time is not None and (__import__('time').time() - t_start) > (wall_time - 5) * 60:
                print(f'\nApproaching wall time limit ({wall_time:.0f} min), stopping gracefully.')
                break
            print(f'\n── {output} ──')

            simfn = pjoin(pynbody_path, output)
            try:
                snap = pynbody.load(simfn)
                snap.physical_units()
                print(f'  Loaded ({len(snap.dm)} DM particles)')
            except Exception as e:
                print(f'  Failed to load snapshot: {e}, skipping')
                continue

            # ── SEED at z=0 (first snapshot only) ────────────────────────────
            if not active_branches:
                if output != outputs_reversed[0]:
                    # Seeding failed at z=0 — don't retry on earlier snaps
                    print(f'  Seed not established at z=0, skipping')
                    del snap
                    continue

                # The halo catalogue is only needed to look up the seed halo.
                # Build it here inside the seed branch rather than once per
                # snapshot — subsequent snaps never use it (they build the HOP
                # catalogue later for the halonum/r200 lookup), so constructing
                # it every loop was wasted work.
                try:
                    halo_cat = load_halo_catalogue(snap, use_ahf)
                except Exception as e:
                    print(f'  Failed to load halo catalogue: {e}, skipping')
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

                # For seeding, clip to 1×r200 — the halo is centred at origin
                # and the wide search sphere is only needed for satellites / tracking
                seed_dist = np.sqrt((positions ** 2).sum(axis=1))
                seed_mask = seed_dist <= r200
                seed_pos   = positions[seed_mask]
                seed_iords = iords_all[seed_mask]

                mask = _density_region_grow(
                    seed_pos, seed_iords, voxel_size,
                    prev_iords=None, min_cluster_size=min_cluster_size,
                )
                if mask is None or mask.sum() == 0:
                    print(f'  Seed clustering failed, skipping')
                    continue

                main_iords    = seed_iords[mask]
                main_pos      = seed_pos[mask]
                centroid      = main_pos.mean(axis=0)
                bounding_r    = float(np.sqrt(((main_pos - centroid)**2).sum(axis=1)).max())

                # Find corresponding HOP halo for tangos merger tree queries
                hop_hnum = find_hop_halonum(snap, main_iords)

                active_branches['main'] = {
                    'prev_iords':    main_iords,
                    'prev_halonum':  halonumber,
                    'prev_centroid': centroid,
                }

                grp = h5f.require_group(output)
                mg  = grp.require_group('main')
                mg.create_dataset('iords',            data=main_iords.astype(np.int64))
                mg.create_dataset('halonum',          data=np.int64(halonumber))
                mg.create_dataset('halonum_reliable', data=np.bool_(True))
                if hop_hnum is not None:
                    mg.create_dataset('hop_halonum',  data=np.int64(hop_hnum))
                mg.create_dataset('centroid',         data=centroid.astype(np.float64))
                mg.create_dataset('bounding_radius',  data=np.float64(bounding_r))
                h5f.flush()
                print(f'  Seeded main branch: {len(main_iords)} particles '
                      f'(halo {halonumber}, hop={hop_hnum}, r200={r200:.2f} kpc, '
                      f'bounding_r={bounding_r:.2f} kpc)')
                del snap, dm_within
                continue

            # ── SUBSEQUENT SNAPS ──────────────────────────────────────────────

            prev_iords = active_branches['main']['prev_iords']

            # Step 1: find prev_iords in this snapshot, center on them
            all_dm_iords = np.array(snap.dm['iord'])
            in_prev = np.isin(all_dm_iords, prev_iords)
            if not in_prev.any():
                print(f'  No previous particles found in snapshot, skipping')
                del snap
                continue

            prev_pos = np.array(snap.dm['pos'])[in_prev]
            centroid_prev = prev_pos.mean(axis=0)
            snap.dm['pos'] -= centroid_prev  # center on previous cluster

            # Step 2: find HOP halo with most overlap with prev_iords.
            # Build the HOP catalogue ONCE and reuse it for both the halonum
            # search and the r200 lookup below. Constructing it runs the HOP
            # group finder and is expensive; it was previously built twice per
            # snapshot (once inside find_hop_halonum, once again for r200).
            try:
                hop_cat = pynbody.halo.hop.HOPCatalogue(snap)
            except Exception as e:
                print(f'  Could not load HOP catalogue: {e}')
                hop_cat = None

            hop_hnum = find_hop_halonum(snap, prev_iords, hop_cat=hop_cat)
            hop_halonum = hop_hnum if hop_hnum is not None else active_branches['main'].get('prev_hop_halonum')

            # Get r200 from the HOP halo if found, else use bounding_r as fallback
            r200_main = None
            if hop_hnum is not None and hop_cat is not None:
                try:
                    h_hop = hop_cat[hop_hnum - 1]  # 0-indexed
                    r200_main = get_r200(h_hop)
                except Exception:
                    pass
            if r200_main is None or r200_main <= 0:
                prev_bounding = active_branches['main'].get('prev_bounding_r', 20.0)
                r200_main = prev_bounding * 2.0
                print(f'  Using fallback r200={r200_main:.1f} kpc from prev bounding_r')

            # Step 3: take particles within search_rad * r200
            pos  = np.array(snap.dm['pos'])
            dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
            within_mask = dist <= search_rad * r200_main
            positions = pos[within_mask]
            iords_all = all_dm_iords[within_mask]

            if len(iords_all) == 0:
                print(f'  No DM particles within search radius, skipping')
                del snap
                continue

            print(f'  r200={r200_main:.1f} kpc, {len(iords_all)} DM in search sphere, hop={hop_halonum}')

            # Step 4: HDBSCAN on voxels to remove satellites
            mask = _density_region_grow(
                positions, iords_all, voxel_size,
                prev_iords=prev_iords,
                min_cluster_size=min_cluster_size,
            )

            if mask is None or mask.sum() == 0:
                print(f'  No cluster found, skipping snap')
                del snap
                continue

            cluster_iords = iords_all[mask]
            cluster_pos   = positions[mask]
            centroid      = cluster_pos.mean(axis=0)
            bounding_r    = float(np.sqrt(((cluster_pos - centroid)**2).sum(axis=1)).max())

            # Step 5: store results
            active_branches['main']['prev_iords']      = cluster_iords
            active_branches['main']['prev_centroid']    = centroid
            active_branches['main']['prev_bounding_r']  = bounding_r
            if hop_halonum is not None:
                active_branches['main']['prev_hop_halonum'] = hop_halonum

            grp = h5f.require_group(output)
            mg  = grp.require_group('main')
            mg.create_dataset('iords',           data=cluster_iords.astype(np.int64))
            mg.create_dataset('centroid',        data=centroid.astype(np.float64))
            mg.create_dataset('bounding_radius', data=np.float64(bounding_r))
            if hop_halonum is not None:
                mg.create_dataset('hop_halonum', data=np.int64(hop_halonum))

            print(f'  Main branch: {len(cluster_iords)} particles '
                  f'(hop={hop_halonum}, r200={r200_main:.1f} kpc, '
                  f'bounding_r={bounding_r:.2f} kpc)')

            h5f.flush()
            del snap

    finally:
        h5f.close()

    print(f'\nDone. Written to {output_path}')
    print(f'Total branches: {len(active_branches)}')


if __name__ == '__main__':
    main()
