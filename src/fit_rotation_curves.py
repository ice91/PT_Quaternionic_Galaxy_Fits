#!/usr/bin/env python
"""
Batch-fit SPARC galaxies with Quaternionic, ΛCDM & MOND models.

Examples
--------
python -m src.fit_rotation_curves --data-path data/xx_sparc.dat --outdir results/
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from tqdm import tqdm

from .models import v_quaternionic, v_lcdm, v_mond
from .constants import ArrayLike

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _chi2(y_obs: ArrayLike, y_mod: ArrayLike, err: ArrayLike) -> float:
    return np.sum(((y_obs - y_mod) / err) ** 2)


def _aic(chi2: float, k: int) -> float:
    return chi2 + 2 * k


def _bic(chi2: float, k: int, n: int) -> float:
    return chi2 + k * np.log(n)


def _percentile_ci(samples: ArrayLike, p=(16, 84)):
    return [np.percentile(samples[:, i], p) for i in range(samples.shape[1])]


# ----------------------------------------------------------------------
# Log-posteriors
# ----------------------------------------------------------------------
def _lp_quat(theta):
    m_star, r_s, eps = theta
    if 1e8 < m_star < 1e11 and 0.1 < r_s < 20 and 0 < eps < 1e4:
        return 0.0
    return -np.inf


def _lg_quat(theta, r, v, err):
    return -0.5 * _chi2(v, v_quaternionic(r, *theta), err)


def _lnpost_quat(theta, r, v, err):
    lp = _lp_quat(theta)
    return lp + _lg_quat(theta, r, v, err) if np.isfinite(lp) else -np.inf


# LCDM
def _lp_lcdm(theta):
    m_star, r_s, m_dm, r_s_dm = theta
    if 1e8 < m_star < 1e11 and 0.1 < r_s < 20 and 1e8 < m_dm < 1e12 and 1 < r_s_dm < 50:
        return 0.0
    return -np.inf


def _lg_lcdm(theta, r, v, err):
    return -0.5 * _chi2(v, v_lcdm(r, *theta), err)


def _lnpost_lcdm(theta, r, v, err):
    lp = _lp_lcdm(theta)
    return lp + _lg_lcdm(theta, r, v, err) if np.isfinite(lp) else -np.inf


# MOND
def _lp_mond(theta):
    m_star, r_s, g_ext = theta
    if 1e8 < m_star < 1e11 and 0.1 < r_s < 20 and 0 < g_ext < 1e-9:
        return 0.0
    return -np.inf


def _lg_mond(theta, r, v, err):
    return -0.5 * _chi2(v, v_mond(r, *theta), err)


def _lnpost_mond(theta, r, v, err):
    lp = _lp_mond(theta)
    return lp + _lg_mond(theta, r, v, err) if np.isfinite(lp) else -np.inf


# ----------------------------------------------------------------------
# Main fitter
# ----------------------------------------------------------------------
def run_fit(df: pd.DataFrame,
            outdir: Path,
            nwalkers: int = 32,
            steps: int = 2000):
    outdir.mkdir(parents=True, exist_ok=True)
    galaxies = df['galaxy_id'].unique()
    results = []

    for gid in tqdm(galaxies, desc="Galaxies"):
        sub = df[df['galaxy_id'] == gid].dropna()
        if len(sub) < 4:
            continue

        r_kpc = sub['rad[kpc]'].values
        v_obs = sub['vobs[km/s]'].values
        v_err = np.clip(sub['errv[km/s]'].values, 0.1, None)
        m_star_guess = sub['mass[M_sun]'].iloc[0]

        # ---------------- Quaternionic ----------------
        ndim_q = 3
        p0_q = np.array([m_star_guess, 3.0, 0.05]) * (1 + 0.1 * np.random.randn(nwalkers, ndim_q))
        sam_q = emcee.EnsembleSampler(nwalkers, ndim_q, _lnpost_quat,
                                      args=(r_kpc, v_obs, v_err))
        sam_q.run_mcmc(p0_q, steps, progress=False)
        chain_q = sam_q.get_chain(discard=steps // 4, flat=True)
        m_star_q, r_s_q, eps_q = np.median(chain_q, axis=0)
        ci_q = _percentile_ci(chain_q)
        v_q = v_quaternionic(r_kpc, m_star_q, r_s_q, eps_q)
        chi2_q = _chi2(v_obs, v_q, v_err)

        # ---------------- ΛCDM ----------------
        ndim_l = 4
        p0_l = np.array([m_star_guess, 3.0, m_star_guess * 5, 15.0]) * \
               (1 + 0.1 * np.random.randn(nwalkers, ndim_l))
        sam_l = emcee.EnsembleSampler(nwalkers, ndim_l, _lnpost_lcdm,
                                      args=(r_kpc, v_obs, v_err))
        sam_l.run_mcmc(p0_l, steps, progress=False)
        chain_l = sam_l.get_chain(discard=steps // 4, flat=True)
        m_star_l, r_s_l, m_dm_l, r_s_dm_l = np.median(chain_l, axis=0)
        ci_l = _percentile_ci(chain_l)
        v_l = v_lcdm(r_kpc, m_star_l, r_s_l, m_dm_l, r_s_dm_l)
        chi2_l = _chi2(v_obs, v_l, v_err)

        # ---------------- MOND ----------------
        ndim_m = 3
        p0_m = np.array([m_star_guess, 3.0, 1e-10]) * \
               (1 + 0.1 * np.random.randn(nwalkers, ndim_m))
        sam_m = emcee.EnsembleSampler(nwalkers, ndim_m, _lnpost_mond,
                                      args=(r_kpc, v_obs, v_err))
        sam_m.run_mcmc(p0_m, steps, progress=False)
        chain_m = sam_m.get_chain(discard=steps // 4, flat=True)
        m_star_m, r_s_m, g_ext_m = np.median(chain_m, axis=0)
        ci_m = _percentile_ci(chain_m)
        v_m = v_mond(r_kpc, m_star_m, r_s_m, g_ext_m)
        chi2_m = _chi2(v_obs, v_m, v_err)

        # Information criteria
        AIC_q, BIC_q = _aic(chi2_q, ndim_q), _bic(chi2_q, ndim_q, len(v_obs))
        AIC_l, BIC_l = _aic(chi2_l, ndim_l), _bic(chi2_l, ndim_l, len(v_obs))
        AIC_m, BIC_m = _aic(chi2_m, ndim_m), _bic(chi2_m, ndim_m, len(v_obs))

        best = min({'Quaternionic': AIC_q, 'LCDM': AIC_l, 'MOND': AIC_m},
                   key=lambda k: {'Quaternionic': AIC_q, 'LCDM': AIC_l, 'MOND': AIC_m}[k])

        results.append(dict(
            Galaxy=gid,
            chi2_Quat=chi2_q, AIC_Quat=AIC_q, BIC_Quat=BIC_q,
            M_solar_Quat=m_star_q, rs_kpc_Quat=r_s_q, epsilon_Quat=eps_q,
            chi2_LCDM=chi2_l, AIC_LCDM=AIC_l, BIC_LCDM=BIC_l,
            chi2_MOND=chi2_m, AIC_MOND=AIC_m, BIC_MOND=BIC_m,
            Best_Model=best
        ))

        # ---- Plot ----
        plt.figure(figsize=(8, 5))
        plt.errorbar(r_kpc, v_obs, yerr=v_err, fmt='o', label='Data', alpha=0.6)
        plt.plot(r_kpc, v_q, label='Quaternionic')
        plt.plot(r_kpc, v_l, label='ΛCDM')
        plt.plot(r_kpc, v_m, label='MOND')
        plt.xscale('log')
        plt.xlabel('Radius [kpc]')
        plt.ylabel('v [km/s]')
        plt.title(f'{gid}')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(outdir / f'{gid}.png', dpi=150)
        plt.close()

    # --------- Summary CSV ---------
    df_out = pd.DataFrame(results)
    df_out.to_csv(outdir / 'sparc_rotation_curve_comparison_all.csv', index=False)
    print(f"\nSaved summary to {outdir/'sparc_rotation_curve_comparison_all.csv'}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fit SPARC rotation curves.")
    p.add_argument('--data-path', required=True,
                   help='Path to SPARC ASCII file.')
    p.add_argument('--outdir', default='results/',
                   help='Directory to save plots & CSV summary.')
    p.add_argument('--nwalkers', type=int, default=32,
                   help='Number of MCMC walkers.')
    p.add_argument('--steps', type=int, default=2000,
                   help='Steps per walker.')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    df = pd.read_csv(args.data_path, sep=r'\s+', names=[
        'galaxy_id', 'rad[kpc]', 'vobs[km/s]', 'errv[km/s]', 'mass[M_sun]'])
    run_fit(df, Path(args.outdir), args.nwalkers, args.steps)


if __name__ == "__main__":
    main()
