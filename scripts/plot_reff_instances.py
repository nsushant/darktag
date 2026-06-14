"""
Aggregate reff_instance_*.csv files and plot mean ± 1σ confidence interval.

Writes a summary CSV and (optionally) a plot.

Usage:
    python scripts/plot_reff_instances.py <reffs_dir>
    python scripts/plot_reff_instances.py <reffs_dir> --output-csv out.csv --output-plot out.png
    python scripts/plot_reff_instances.py <reffs_dir> --no-plot
"""

import sys
import os
import glob
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-instance reff CSVs and plot confidence intervals'
    )
    parser.add_argument('reffs_dir', help='Directory containing reff_instance_*.csv files')
    parser.add_argument('--output-csv', default=None,
                        help='Output summary CSV path (default: <reffs_dir>/summary.csv)')
    parser.add_argument('--output-plot', default=None,
                        help='Output plot path (default: <reffs_dir>/summary.png)')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Skip plotting, only write summary CSV')
    args = parser.parse_args()

    reffs_dir   = args.reffs_dir.rstrip('/')
    output_csv  = args.output_csv  or os.path.join(reffs_dir, 'summary.csv')
    output_plot = args.output_plot or os.path.join(reffs_dir, 'summary.png')

    # ── Load instance files ───────────────────────────────────────────────────
    pattern = os.path.join(reffs_dir, 'reff_instance_*.csv')
    fnames  = sorted(glob.glob(pattern))
    if len(fnames) == 0:
        print(f'Error: no reff_instance_*.csv files found in {reffs_dir}')
        sys.exit(1)
    print(f'Found {len(fnames)} instance files')

    dfs = []
    for f in fnames:
        df = pd.read_csv(f, index_col=0)
        dfs.append(df[['t', 'z', 'reff', 'halflight']])

    combined = pd.concat(dfs)

    # ── Aggregate by t ────────────────────────────────────────────────────────
    grp = combined.groupby('t')

    summary = pd.DataFrame({
        't':              grp['t'].mean(),
        'z':              grp['z'].mean(),
        'mean_reff':      grp['reff'].mean(),
        'std_reff_plus':  grp['reff'].mean() + grp['reff'].std(),
        'std_reff_minus': (grp['reff'].mean() - grp['reff'].std()).clip(lower=0),
        'mean_halflight': grp['halflight'].mean(),
        'std_halflight_plus':  grp['halflight'].mean() + grp['halflight'].std(),
        'std_halflight_minus': (grp['halflight'].mean() - grp['halflight'].std()).clip(lower=0),
        'n_instances':    grp['reff'].count(),
    }).reset_index(drop=True).sort_values('t')

    summary.to_csv(output_csv, index=False)
    print(f'Wrote summary CSV: {output_csv}')

    if args.no_plot:
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    t   = summary['t'].values
    mu  = summary['mean_reff'].values
    hi  = summary['std_reff_plus'].values
    lo  = summary['std_reff_minus'].values

    ax.plot(t, mu, color='steelblue', lw=2, label='mean $R_\\mathrm{eff}$')
    ax.fill_between(t, lo, hi, color='steelblue', alpha=0.3, label='$\\pm 1\\sigma$')

    ax.set_xlabel('$t$ (Gyr)')
    ax.set_ylabel('$R_\\mathrm{eff}$ (kpc)')
    ax.legend()
    ax.set_title(os.path.basename(reffs_dir))

    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    print(f'Wrote plot: {output_plot}')


if __name__ == '__main__':
    main()
