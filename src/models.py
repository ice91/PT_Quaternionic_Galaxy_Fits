"""
Analytical rotation-curve models used in Chen (2025).
"""

from __future__ import annotations
import numpy as np
from .constants import G, M_SUN, KPC_TO_M, A0_MOND, ArrayLike


# -------------------------------------------------------------
# 1. Quaternionic (simplified) model
# -------------------------------------------------------------
def v_quaternionic(r_kpc: ArrayLike,
                   m_star: float,
                   r_s_kpc: float,
                   epsilon: float) -> ArrayLike:
    """
    Quaternionic rotation curve (simplified, no 0.5 prefactor).

    Parameters
    ----------
    r_kpc   : galactocentric radius [kpc]
    m_star  : baryonic mass          [M_sun]
    r_s_kpc : exponential scale-length [kpc]
    epsilon : dimensionless geometric-flow coupling

    Returns
    -------
    v_kms : rotation velocity [km/s]
    """
    r_m  = r_kpc * KPC_TO_M
    rs_m = r_s_kpc * KPC_TO_M

    # Baryonic contribution (exponential disk)
    v_baryon = np.sqrt(G * m_star * M_SUN * (1 - np.exp(-r_m / rs_m)) / r_m)

    # Geometry-flow correction
    correction = epsilon * (r_m / rs_m)

    return np.sqrt(v_baryon ** 2 + correction) * 1e-3  # km/s


# -------------------------------------------------------------
# 2. ΛCDM (NFW-like) model
# -------------------------------------------------------------
def v_lcdm(r_kpc: ArrayLike,
           m_star: float,
           r_s_kpc: float,
           m_dm: float,
           r_s_dm_kpc: float) -> ArrayLike:
    r_m = r_kpc * KPC_TO_M
    v_baryon = np.sqrt(G * m_star * M_SUN / r_m)
    v_dm = np.sqrt(G * m_dm * M_SUN *
                   (1 - np.exp(-r_m / (r_s_dm_kpc * KPC_TO_M))) / r_m)
    return np.sqrt(v_baryon ** 2 + v_dm ** 2) * 1e-3  # km/s


# -------------------------------------------------------------
# 3. MOND (simple interpolating function + external field)
# -------------------------------------------------------------
def v_mond(r_kpc: ArrayLike,
           m_star: float,
           r_s_kpc: float,
           g_ext: float) -> ArrayLike:
    r_m  = r_kpc * KPC_TO_M
    v_N  = np.sqrt(G * m_star * M_SUN / r_m)
    a_N  = v_N ** 2 / r_m
    mu   = a_N / (A0_MOND + g_ext)
    v_M  = v_N * np.sqrt(0.5 + 0.5 * np.sqrt(1 + 4 / mu ** 2))
    return v_M * 1e-3  # km/s
