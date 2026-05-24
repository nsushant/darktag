"""
Cross-reference AHF halo numbers with tangos/HOP halos
by matching DM particle iords across snapshots.

Usage:
    python scripts/crossref_ahf_halonums.py Halo1459_DMO
    python scripts/crossref_ahf_halonums.py Halo1459_DMO --max-ahf 500 --mass-tol 0.3 -o ahf_halonums.csv
"""

import sys
import os
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config
from darktag.tagging.clustering import cluster_tagged_particles


def load_indexing_data(DMOsim, halo_number):
    main_halo = DMOsim.timesteps[-1].halos[int(halo_number) - 1]

    halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]

    t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
    red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]

    outputs = np.array([
        DMOsim.timesteps[i].__dict__['extension']
        for i in range(len(DMOsim.timesteps))
    ])
    times_tangos = np.array([
        DMOsim.timesteps[i].__dict__['time_gyr']
        for i in range(len(DMOsim.timesteps))
    ])

    valid_outputs = outputs[np.isin(times_tangos, t_all)]
    valid_outputs.sort()

    return t_all, red_all, main_halo, halonums, valid_outputs


def check_ahf_halo_mass(args):
    key, ahf_cat, prev_iords, expected_m200, mass_tol = args
    try:
        h = ahf_cat[key]
        total_mass = h.dm['mass'].sum()
        if expected_m200 is not None:
            if total_mass < expected_m200 * mass_tol or total_mass > expected_m200 / mass_tol:
                return key, 0, total_mass
        iords = h.dm['iord']
        overlap = np.isin(iords, prev_iords).sum()
        return key, overlap, total_mass
    except Exception:
        return key, 0, 0


def mass_filtered_search(ahf_cat, prev_iords, max_ahf,
                         expected_m200=None, mass_tol=0.3,
                         parallel=False, num_workers=None):
    keys = sorted(ahf_cat.keys())[:max_ahf]
    if not keys:
        return -1, 0, 0

    task_args = [(k, ahf_cat, prev_iords, expected_m200, mass_tol) for k in keys]

    if parallel and len(keys) > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.map(check_ahf_halo_mass, task_args)
    else:
        results = [check_ahf_halo_mass(a) for a in task_args]

    best = -1
    best_overlap = 0
    best_mass = 0
    for key, overlap, mass in results:
        if overlap > best_overlap:
            best_overlap = overlap
            best = key
            best_mass = mass

    return best, best_overlap, best_mass


def main():
    import pynbody
    from os.path import join as pjoin
    import tangos

    parser = argparse.ArgumentParser(
        description='Cross-reference AHF halo numbers with tangos/HOP halos'
    )
    parser.add_argument('sim_name', help='Tangos simulation name (e.g. Halo1459_DMO)')
    parser.add_argument('--max-ahf', type=int, default=500,
                        help='Max AHF halos to search per snapshot (default: 500)')
    parser.add_argument('--mass-tol', type=float, default=0.3,
                        help='Mass tolerance for AHF halo filtering (default: 0.3)')
    parser.add_argument('--parallel', action='store_true',
                        help='Parallelise AHF halo search')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV path (default: <pynbody_path>/<sim_name>/ahf_halonums.csv)')
    parser.add_argument('--r200-fraction', type=float, default=0.05,
                        help='Fraction of R200 for clustering (default: 0.05)')
    parser.add_argument('--cluster-max-particles', type=int, default=10000,
                        help='Max particles for clustering (random downsample if exceeded, default: 10000)')
    parser.add_argument('--eps', type=float, default=None,
                        help='DBSCAN eps (overrides config, scaled features)')
    args = parser.parse_args()

    sim_name = args.sim_name
    max_ahf = args.max_ahf
    mass_tol = args.mass_tol
    r200_frac = args.r200_fraction
    cluster_max_particles = args.cluster_max_particles
    eps_override = args.eps
    parallel = args.parallel
    num_workers = args.num_workers or mp.cpu_count()

    tangos_path = config.get_path('tangos_path')
    pynbody_path = pjoin(config.get_path('pynbody_path'), sim_name)

    print(f'Initialising tangos DB: {pjoin(tangos_path, sim_name.split("_")[0] + ".db")}')
    tangos.core.init_db(pjoin(tangos_path, sim_name.split('_')[0] + '.db'))
    sim = tangos.get_simulation(sim_name)

    t_all, red_all, main_halo, halonums, outputs = load_indexing_data(sim, 1)
    print(f'Found {len(outputs)} snapshots')

    try:
        main_halo_end = sim.timesteps[-1].halos[int(halonums[-1]) - 1]
        expected_m200 = float(main_halo_end['M200c'])
        print(f'Expected M200 from tangos: {expected_m200:.3e} Msun')
    except Exception:
        expected_m200 = None
        print('Could not get M200 from tangos, skipping mass filter')

    cluster_config = config.get('tagging', 'clustering')
    cluster_kwargs = dict(
        method=cluster_config.get('method', 'hdbscan'),
        feature_cols=cluster_config.get('features', ['x', 'y']),
        scale=True,
    )
    if cluster_kwargs['method'] == 'hdbscan':
        cluster_kwargs.update(dict(
            min_cluster_size=config.get_with_default('hdbscan', 'min_cluster_size', 10),
            hdbscan_min_samples=config.get_with_default('hdbscan', 'min_samples', None),
            cluster_selection_epsilon=config.get_with_default('hdbscan', 'cluster_selection_epsilon', 0.0),
            cluster_selection_method=config.get_with_default('hdbscan', 'cluster_selection_method', 'eom'),
            allow_single_cluster=config.get_with_default('hdbscan', 'allow_single_cluster', True),
            max_cluster_size=config.get_with_default('hdbscan', 'max_cluster_size', None),
        ))
    elif cluster_kwargs['method'] == 'dbscan':
        cluster_kwargs.update(dict(
            eps=eps_override if eps_override is not None else config.get_with_default('dbscan', 'eps', 0.05),
            dbscan_min_samples=config.get_with_default('dbscan', 'min_samples', 2),
        ))

    results = []
    prev_iords = None

    print(f'\n{"="*60}')
    print(f'Phase 1: Bootstrap at z=0 ({outputs[-1]})')
    print(f'{"="*60}')

    s = pynbody.load(pjoin(pynbody_path, outputs[-1]))

    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    h_hop = s.halos()[int(halonums[-1]) - 1]
    s.physical_units()
    pynbody.analysis.halo.center(h_hop.dm)

    try:
        r200 = pynbody.analysis.halo.virial_radius(
            h_hop.d, overden=200, r_max=None, rho_def='critical'
        )
        dm_all = h_hop.dm[h_hop.dm['r'] < r200]
        dm_cluster = dm_all[dm_all['r'] < r200 * r200_frac]
        print(f'  R200 = {r200:.3f} kpc, {len(dm_all)} DM within R200, '
              f'{len(dm_cluster)} for clustering (r < {r200*r200_frac:.2f} kpc)')
    except Exception as e:
        dm_all = h_hop.dm
        dm_cluster = dm_all
        print(f'  R200 failed ({e}), using all HOP DM ({len(dm_all)} particles)')

    n_cluster = len(dm_cluster)
    if n_cluster > cluster_max_particles:
        idx = np.random.choice(n_cluster, cluster_max_particles, replace=False)
        dm_input = dm_cluster[idx]
        print(f'  Downsampled {n_cluster} -> {cluster_max_particles} for clustering')
    else:
        dm_input = dm_cluster

    labels, best_label, _ = cluster_tagged_particles(particles=dm_input, **cluster_kwargs)

    if best_label != -1:
        cluster_iords = np.asarray(dm_input['iord'][labels == best_label])
        prev_iords = np.asarray(dm_all['iord'][np.isin(dm_all['iord'], cluster_iords)])
        print(f'  Clustering: {cluster_kwargs["method"]} cluster {best_label}, '
              f'{len(cluster_iords)} in cluster, expanded to {len(prev_iords)} from R200')
    else:
        prev_iords = np.asarray(dm_all['iord'])
        print(f'  Clustering: no cluster found, using all {len(prev_iords)} particles')

    del s
    s = pynbody.load(pjoin(pynbody_path, outputs[-1]))
    s.physical_units()

    pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
    cat = s.halos(halo_numbers='v1')

    best_ahf, overlap, mass = mass_filtered_search(
        cat, prev_iords, max_ahf,
        expected_m200=expected_m200, mass_tol=mass_tol,
        parallel=parallel, num_workers=num_workers,
    )

    if best_ahf == -1:
        print('  ERROR: No AHF halo matched at bootstrap!')
        sys.exit(1)

    print(f'  Best AHF halo: {best_ahf} (overlap={overlap}, mass={mass:.3e})')
    results.append({'snapshot': outputs[-1], 'AHF halonum': best_ahf})

    del s

    print(f'\n{"="*60}')
    print(f'Phase 2: Tracking backwards through {len(outputs) - 1} snapshots')
    print(f'  mass_tol={mass_tol}, max_ahf={max_ahf}, parallel={parallel}')
    print(f'{"="*60}')

    for snap in reversed(outputs[:-1]):
        print(f'\n  --- {snap} ---')

        try:
            s = pynbody.load(pjoin(pynbody_path, snap))
            s.physical_units()
        except Exception as e:
            print(f'  Could not load ({e}), skipping')
            continue

        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
        cat = s.halos(halo_numbers='v1')

        best_ahf, overlap, mass = mass_filtered_search(
            cat, prev_iords, max_ahf,
            expected_m200=expected_m200, mass_tol=mass_tol,
            parallel=parallel, num_workers=num_workers,
        )

        if best_ahf == -1:
            print(f'  No AHF halo matched, skipping')
            continue

        h = cat[best_ahf]
        pynbody.analysis.halo.center(h.dm)

        try:
            r200 = pynbody.analysis.halo.virial_radius(
                h.d, overden=200, r_max=None, rho_def='critical'
            )
            dm_all = h.dm[h.dm['r'] < r200]
            dm_cluster = dm_all[dm_all['r'] < r200 * r200_frac]
            r200_str = f'r<{r200*r200_frac:.2f} kpc'
        except Exception:
            dm_all = h.dm
            dm_cluster = dm_all
            r200_str = 'all DM (R200 failed)'

        n_cluster = len(dm_cluster)
        if n_cluster > cluster_max_particles:
            idx = np.random.choice(n_cluster, cluster_max_particles, replace=False)
            dm_input = dm_cluster[idx]
        else:
            dm_input = dm_cluster

        labels, best_label, _ = cluster_tagged_particles(
            particles=dm_input, prev_iords=prev_iords, **cluster_kwargs
        )

        if best_label != -1:
            cluster_iords = np.asarray(dm_input['iord'][labels == best_label])
            prev_iords = np.asarray(dm_all['iord'][np.isin(dm_all['iord'], cluster_iords)])
            n = len(prev_iords)
        else:
            n = 0

        print(f'  AHF halo {best_ahf} (overlap={overlap}, mass={mass:.3e}), '
              f'{cluster_kwargs["method"]} cluster {best_label} ({n} particles'
              f' from {len(dm_input)} input within {r200_str})')
        results.append({'snapshot': snap, 'AHF halonum': best_ahf})

    df = pd.DataFrame(results)
    df = df.iloc[::-1].reset_index(drop=True)

    output_path = args.output or pjoin(pynbody_path, 'ahf_halonums.csv')
    df.to_csv(output_path, index=False)

    print(f'\n{"="*60}')
    print(f'Saved: {output_path}')
    print(f'{"="*60}')
    print(df.to_string())


if __name__ == '__main__':
    mp.freeze_support()
    main()
