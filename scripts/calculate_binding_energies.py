"""
Calculate and cache binding energies for DM particles within r200c of the main halo.

Reads halonums per snapshot from a track file, loads the halo, computes PE via
pynbody.gravity.direct (DM-only potential) + KE from the snapshot, and writes
results to an output HDF5 file.

Track file: prefers the per-snapshot 'hop_halonum' written by
scripts/ahf_to_hop_conversion.py (indexes the HOP catalogue); falls back to a
legacy AHF 'halonum'. The catalogue is chosen automatically to match.

Works for both DMO and HYDRO simulations (HYDRO computes the DM binding energy
using the DM-only potential, same as DMO). Resume-safe: skips snapshots already
written.

Output HDF5 structure:
    /output_0089/iords          → int64[N]   particle IDs within r200c
    /output_0089/total_energy   → float64[N] PE + KE per particle (in KE units)
    /output_0089/pe             → float64[N] gravitational potential energy
    /output_0089/ke             → float64[N] kinetic energy
    /output_0089/rank           → int64[N]   argsort rank (0 = most bound)
    /output_0089/r200c          → float64    virial radius in kpc

Usage:
    # DMO, using a HOP-converted track file
    python scripts/calculate_binding_energies.py Halo1459_DMO_Mreionx12 \\
        --track Halo1459_DMO_Mreionx12_hop_track.hdf5 --dmo --wall-time 120

    # HYDRO dark matter, using a HOP-converted track file
    python scripts/calculate_binding_energies.py Halo1459_HYDRO_Mreionx12 \\
        --track Halo1459_HYDRO_Mreionx12_hop_track.hdf5 --wall-time 120
"""

import sys
import os
import glob
from os.path import join as pjoin

import numpy as np

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


def compute_be_for_particles(particles_r200, softening_pc=10.0):
    """
    Compute PE + KE for particles already selected within r200c and centred.

    Returns
    -------
    iords        : int64 array
    total_energy : float64 array  (PE + KE, in KE units)
    pe_arr       : float64 array
    ke_arr       : float64 array
    rank         : int64 array    (argsort of total_energy, 0 = most bound)
    """
    import pynbody

    if len(particles_r200) == 0:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty, empty, np.array([], dtype=np.int64)

    softening = pynbody.array.SimArray(
        np.ones(len(particles_r200)) * softening_pc, units='pc', sim=None
    )

    pe, _ = pynbody.gravity.direct(
        particles_r200, np.asarray(particles_r200['pos']), eps=softening
    )
    ke = particles_r200['ke']

    pe_in_ke_units = np.asarray(pe.in_units(ke.units))
    ke_arr         = np.asarray(ke)
    total_energy   = pe_in_ke_units + ke_arr

    rank = np.argsort(total_energy.flatten()).astype(np.int64)

    return (
        np.asarray(particles_r200['iord'], dtype=np.int64),
        total_energy.astype(np.float64),
        pe_in_ke_units.astype(np.float64),
        ke_arr.astype(np.float64),
        rank,
    )


def main():
    import argparse
    import h5py
    import pynbody
    import pynbody.halo.ahf
    import pynbody.halo.hop
    import time

    parser = argparse.ArgumentParser(
        description='Cache DM particle binding energies within r200c per snapshot'
    )
    parser.add_argument('sim_name',
                        help='Simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--track', required=True,
                        help='Path to track_cluster HDF5 file (provides halonums)')
    parser.add_argument('--output', default=None,
                        help='Output HDF5 path (default: <sim_name>_binding_energies.hdf5)')
    parser.add_argument('--ahf', action='store_true',
                        help='Use AHF halo catalogue (recommended)')
    parser.add_argument('--dmo', action='store_true',
                        help='Use DMO pynbody_path from config (default: hydro_pynbody_path)')
    parser.add_argument('--softening', type=float, default=10.0,
                        help='Softening length in pc for direct gravity (default: 10)')
    parser.add_argument('--r200-fraction', type=float, default=0.1,
                        help='Fraction of r200c to use for BE calculation (default: 0.1)')
    parser.add_argument('--wall-time', type=float, default=None,
                        help='Stop this many minutes before wall-clock limit (graceful exit)')
    args = parser.parse_args()

    sim_name    = args.sim_name
    track_path  = args.track
    output_path = args.output or f'{sim_name}_binding_energies.hdf5'
    use_ahf      = args.ahf
    softening    = args.softening
    wall_time    = args.wall_time
    r200_frac    = args.r200_fraction

    # ── Paths ─────────────────────────────────────────────────────────────────
    if args.dmo:
        _base = config.get_path('pynbody_path')
    else:
        _base = (config.get_with_default('paths', 'hydro_pynbody_path', None)
                 or config.get_path('pynbody_path'))
    pynbody_path = pjoin(_base, sim_name)

    snap_paths = sorted(glob.glob(pjoin(pynbody_path, 'output_*')))
    if not snap_paths:
        snap_paths = sorted(glob.glob(pjoin(pynbody_path, '*')))
    all_outputs = [os.path.basename(p) for p in snap_paths]

    if not all_outputs:
        print(f'Error: no snapshots found in {pynbody_path}')
        sys.exit(1)

    # ── Read track file ────────────────────────────────────────────────────────
    if not os.path.isfile(track_path):
        print(f'Error: track file not found: {track_path}')
        sys.exit(1)

    with h5py.File(track_path, 'r') as tf:
        track_snaps = set(tf.keys())
        # Build map: output_name -> halonum (from main branch). Prefer the HOP
        # halonum written by the AHF->HOP conversion; fall back to a legacy AHF
        # 'halonum'. track_is_hop drives catalogue choice + indexing below.
        halonum_map = {}
        track_is_hop = False
        for snap_key in track_snaps:
            grp = tf[snap_key]
            if 'main' not in grp:
                continue
            m = grp['main']
            if 'hop_halonum' in m:
                halonum_map[snap_key] = int(m['hop_halonum'][()])
                track_is_hop = True
            elif 'halonum' in m:
                halonum_map[snap_key] = int(m['halonum'][()])

    print(f"Track file: {track_path}  ({len(halonum_map)} snapshots, "
          f"{'HOP' if track_is_hop else 'AHF'} halonums)")
    print(f'Simulation: {pynbody_path}')
    print(f'Output:     {output_path}')
    print(f'Softening:  {softening} pc')

    # Only process snapshots present in both the sim dir and the track file
    outputs_to_process = [o for o in all_outputs if o in halonum_map]
    if not outputs_to_process:
        print('No overlapping snapshots between sim dir and track file. Exiting.')
        sys.exit(1)

    # Process in reverse time order (z=0 first) to match track_cluster convention,
    # but order doesn't affect results — process high-z → low-z for easier batching.
    outputs_to_process = sorted(outputs_to_process)

    # ── Resume ────────────────────────────────────────────────────────────────
    already_done = set()
    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as hf:
            for key in hf.keys():
                # Only mark as done if the group has all expected datasets
                grp = hf[key]
                if all(d in grp for d in ('iords', 'total_energy', 'pe', 'ke', 'rank', 'r200c', 'r200_frac')):
                    already_done.add(key)
        print(f'Resuming: {len(already_done)} snapshots already complete, '
              f'{len(outputs_to_process) - len(already_done)} remaining')

    # ── Main loop ─────────────────────────────────────────────────────────────
    t_start = time.time()

    with h5py.File(output_path, 'a') as hf:
        for output in outputs_to_process:
            if wall_time is not None:
                elapsed_min = (time.time() - t_start) / 60.0
                if elapsed_min > wall_time - 5:
                    print(f'\nApproaching wall time ({wall_time:.0f} min), stopping gracefully.')
                    break

            if output in already_done:
                print(f'{output}: already done, skipping')
                continue

            halonum = halonum_map[output]
            print(f'\n── {output}  (halo {halonum}) ──')

            simfn = pjoin(pynbody_path, output)
            try:
                snap = pynbody.load(simfn)
                snap.physical_units()
            except Exception as e:
                print(f'  Failed to load snapshot: {e}, skipping')
                continue

            # Load halo catalogue. A HOP track file forces the HOP catalogue
            # (and 1-based hop_halonum indexing); otherwise honour --ahf.
            try:
                if track_is_hop or not use_ahf:
                    pynbody.config['halo-class-priority'] = [pynbody.halo.hop.HOPCatalogue]
                    halo_cat = snap.halos()
                else:
                    pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
                    halo_cat = snap.halos(halo_numbers='v1')
            except Exception as e:
                print(f'  Failed to load halo catalogue: {e}, skipping')
                del snap
                continue

            # Load halo (HOP catalogue is 0-indexed; hop_halonum is 1-based)
            try:
                h = halo_cat[halonum - 1] if track_is_hop else halo_cat[halonum]
            except Exception as e:
                print(f'  Failed to load halo {halonum}: {e}, skipping')
                del snap
                continue

            # Centre and orient
            try:
                pynbody.analysis.halo.center(h)
            except Exception as e:
                print(f'  Centering failed: {e}, skipping')
                del snap
                continue

            # Compute r200c
            try:
                r200c = float(pynbody.analysis.halo.virial_radius(
                    h, overden=200, r_max=None, rho_def='critical'
                ))
            except Exception as e:
                print(f'  virial_radius failed: {e}, skipping')
                del snap
                continue

            if r200c <= 0:
                print(f'  r200c = {r200c:.3f} kpc, skipping')
                del snap
                continue

            # Select DM particles within r200_frac * r200c
            r_cut = r200_frac * r200c
            dm = h.dm
            pos = np.asarray(dm['pos'])
            r   = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
            dm_r200 = dm[r <= r_cut]

            if len(dm_r200) == 0:
                print(f'  No DM particles within {r200_frac}×r200c ({r_cut:.2f} kpc), skipping')
                del snap
                continue

            print(f'  r200c = {r200c:.2f} kpc,  cut = {r200_frac}×r200c = {r_cut:.2f} kpc,  {len(dm_r200)} DM particles')

            # Compute binding energies
            try:
                iords, total_e, pe, ke, rank = compute_be_for_particles(dm_r200, softening)
            except Exception as e:
                print(f'  BE calculation failed: {e}, skipping')
                del snap
                continue

            # Write to HDF5 — delete any partial group first to avoid "name already exists"
            if output in hf:
                del hf[output]
            grp = hf.create_group(output)
            grp.create_dataset('iords',        data=iords)
            grp.create_dataset('total_energy', data=total_e)
            grp.create_dataset('pe',           data=pe)
            grp.create_dataset('ke',           data=ke)
            grp.create_dataset('rank',         data=rank)
            grp.create_dataset('r200c',        data=np.float64(r200c))
            grp.create_dataset('r200_frac',    data=np.float64(r200_frac))
            hf.flush()

            print(f'  Written: {len(iords)} particles  '
                  f'(E_min={total_e[rank[0]]:.4g}, E_max={total_e[rank[-1]]:.4g})')

            del snap, dm, dm_r200

    print(f'\nDone. Results in {output_path}')


if __name__ == '__main__':
    main()
