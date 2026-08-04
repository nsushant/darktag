"""
Compare DarkLight (DMO=False) stellar mass history against the actual hydro
stellar mass read from halo star particles, per snapshot.

Usage:
    python scripts/plot_mstar_comparison.py Halo1459_fiducial_Mreionx12 \
        --track Halo1459_fiducial_Mreionx12_hop_track.hdf5 \
        --db-name Halo1459

    python scripts/plot_mstar_comparison.py Halo1459_fiducial_Mreionx02 \
        --track Halo1459_fiducial_Mreionx02_hop_track.hdf5 \
        --db-name Halo1459
"""

import sys
import os
from os.path import join as pjoin

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~'))
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(os.path.abspath(__file__)), '..')))

from darktag.config import config


def main():
    import argparse
    import h5py
    import pynbody
    import pynbody.halo.hop
    import pynbody.halo.ahf

    parser = argparse.ArgumentParser(
        description='Compare DarkLight vs hydro stellar mass histories')
    parser.add_argument('sim_name', help='Tangos simulation name')
    parser.add_argument('--track', required=True,
                        help='HOP track file (provides halonums per snapshot)')
    parser.add_argument('--db-name', type=str, default=None,
                        help='Tangos DB stem (default: first token of sim_name)')
    parser.add_argument('--n-darklight', type=int, default=5,
                        help='Number of DarkLight realisations to plot (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path (default: <sim_name>_mstar_comparison.png)')
    args = parser.parse_args()

    sim_name = args.sim_name
    tangos_path = config.get_path('tangos_path')
    db_stem = args.db_name or sim_name.split('_')[0]
    pynbody_base = (config.get_with_default('paths', 'hydro_pynbody_path', None)
                    or config.get_path('pynbody_path'))
    pynbody_path = pjoin(pynbody_base, sim_name)
    output_path = args.output or f'{sim_name}_mstar_comparison.png'

    # ── Tangos setup ──────────────────────────────────────────────────────────
    import tangos
    tangos.core.init_db(pjoin(tangos_path, db_stem + '.db'))
    sim = tangos.get_simulation(sim_name)

    main_halo = sim.timesteps[-1].halos[0]
    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    outputs = np.array([
        sim.timesteps[i].__dict__['extension']
        for i in range(len(sim.timesteps))
    ])
    times_tangos = np.array([
        sim.timesteps[i].__dict__['time_gyr']
        for i in range(len(sim.timesteps))
    ])
    valid = np.isin(times_tangos, t_all)
    outputs = outputs[valid]
    t_all = t_all[:len(outputs)]

    # ── DarkLight mass histories ──────────────────────────────────────────────
    from darklight import DarkLight
    print(f'Running DarkLight (DMO=False) {args.n_darklight} times...')
    dl_histories = []
    for k in range(args.n_darklight):
        t_dl, _, _, _, mstar_insitu, _ = DarkLight(main_halo, DMO=False, n=1, mergers=False)
        # mstar_insitu can be 1D or 2D
        arr = np.asarray(mstar_insitu)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        dl_histories.append((np.asarray(t_dl), arr))
    print('  done.')

    # ── Read track file ───────────────────────────────────────────────────────
    halonum_map = {}
    track_is_hop = False
    with h5py.File(args.track, 'r') as tf:
        for snap_key in tf.keys():
            if 'main' not in tf[snap_key]:
                continue
            m = tf[snap_key]['main']
            if 'hop_halonum' in m:
                halonum_map[snap_key] = int(m['hop_halonum'][()])
                track_is_hop = True
            elif 'halonum' in m:
                halonum_map[snap_key] = int(m['halonum'][()])

    print(f'Track file: {len(halonum_map)} snapshots, '
          f"{'HOP' if track_is_hop else 'AHF'} halonums")

    # ── Hydro mstar from halo star particles ──────────────────────────────────
    hydro_t = []
    hydro_mstar = []

    for i, output in enumerate(outputs):
        if output not in halonum_map:
            continue
        hnum = halonum_map[output]

        simfn = pjoin(pynbody_path, output)
        try:
            snap = pynbody.load(simfn)
            snap.physical_units()
        except Exception as e:
            print(f'  {output}: failed to load ({e}), skipping')
            continue

        try:
            if track_is_hop:
                pynbody.config['halo-class-priority'] = [pynbody.halo.hop.HOPCatalogue]
                h = snap.halos()[hnum - 1]
            else:
                pynbody.config['halo-class-priority'] = [pynbody.halo.ahf.AHFCatalogue]
                h = snap.halos(halo_numbers='v1')[hnum]

            pynbody.analysis.halo.center(h)
            try:
                r200 = pynbody.analysis.halo.virial_radius(
                    h.d, overden=200, r_max=None, rho_def='critical')
            except Exception:
                r200 = 30.0  # fallback

            # grab stars from full snapshot within r200
            st = snap.st
            if len(st) > 0:
                r_st = np.sqrt(st['pos'][:, 0]**2 + st['pos'][:, 1]**2 + st['pos'][:, 2]**2)
                st_within = st[r_st <= r200]
                mstar = float(st_within['mass'].sum().in_units('Msol')) if len(st_within) > 0 else 0.0
            else:
                mstar = 0.0
        except Exception as e:
            print(f'  {output}: failed to get halo ({e}), skipping')
            del snap
            continue

        hydro_t.append(t_all[i])
        hydro_mstar.append(mstar)
        print(f'  {output}: t={t_all[i]:.3f} Gyr, mstar={mstar:.0f} Msol')
        del snap

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # DarkLight realisations
    for k, (t_dl, mstar_dl) in enumerate(dl_histories):
        label = 'DarkLight (DMO=False)' if k == 0 else None
        ax.plot(t_dl, mstar_dl, color='C0', alpha=0.3, linewidth=1, label=label)

    # Hydro mstar
    ax.plot(hydro_t, hydro_mstar, 'k-o', markersize=3, linewidth=2,
            label='Hydro h.st mass')

    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel('Stellar Mass [Msol]')
    ax.set_title(f'{sim_name}: DarkLight vs Hydro Mstar')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'\nSaved plot to {output_path}')


if __name__ == '__main__':
    main()
