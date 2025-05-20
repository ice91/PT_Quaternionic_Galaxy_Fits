# src/models.py
import numpy as np
from .constants import G, M_SUN, KPC_TO_M, A0_MOND # 從同級目錄的 constants.py 導入

def rotation_curve_quat_simple(r_kpc, M_solar_val, rs_kpc_val, epsilon_val):
    """
    Corrected Quaternionic Model (simplified):
      v(r) = sqrt[ (G*M_solar*M_sun*(1 - exp(-r_m/rs_m)) / r_m)
                   + (epsilon * (r_m / rs_m)) ]
    Units: r_kpc, rs_kpc in kpc; M_solar in M_sun. Returns km/s.
    epsilon is dimensionless.
    """
    r_m = np.asarray(r_kpc) * KPC_TO_M
    rs_m = rs_kpc_val * KPC_TO_M

    # Handle r_m = 0 to avoid division by zero
    v_baryon_sq = np.zeros_like(r_m, dtype=float)
    non_zero_r_mask = r_m != 0
    if np.any(non_zero_r_mask):
        r_m_nz = r_m[non_zero_r_mask]
        rs_m_eff_nz = rs_m # Assuming rs_m is scalar or same shape as r_m_nz
        if rs_m == 0: # Avoid division by zero in exp if rs_m is globally zero
             exp_term_nz = np.zeros_like(r_m_nz) if rs_m == 0 else np.exp(-r_m_nz / rs_m_eff_nz)
        else:
             exp_term_nz = np.exp(-r_m_nz / rs_m_eff_nz)

        v_baryon_sq[non_zero_r_mask] = (G * M_solar_val * M_SUN * (1 - exp_term_nz)) / r_m_nz

    # Handle rs_m = 0 for the correction term
    correction_term = np.zeros_like(r_m, dtype=float)
    if rs_m != 0:
        correction_term = epsilon_val * (r_m / rs_m)
    else:
        # If rs_m is zero, the definition of correction might need re-evaluation
        # For now, assume it leads to a very large or undefined contribution if r_m > 0
        correction_term[r_m > 0] = np.inf


    v_total_sq = v_baryon_sq + correction_term
    # Ensure velocity squared is not negative due to numerical precision
    v_total_sq[v_total_sq < 0] = 0
    
    return np.sqrt(v_total_sq) * 1e-3  # Convert m/s to km/s

def rotation_curve_LCDM(r_kpc, M_solar_val, rs_kpc_baryon_val, M_dm_val, rs_dm_kpc_val):
    """
    ΛCDM Model:
    v(r) = sqrt[ v_baryon^2 + v_dm^2 ]
    Units: r_kpc, rs_kpc_baryon, rs_dm_kpc in kpc; M_solar, M_dm in M_sun. Returns km/s.
    """
    r_m = np.asarray(r_kpc) * KPC_TO_M
    
    # Baryonic component (point mass approximation for simplicity, or use exponential disk)
    # Using point mass like in your original LCDM for consistency with that version:
    v_baryon_sq = np.zeros_like(r_m, dtype=float)
    non_zero_r_mask = r_m != 0
    if np.any(non_zero_r_mask):
        v_baryon_sq[non_zero_r_mask] = (G * M_solar_val * M_SUN) / r_m[non_zero_r_mask]

    # Dark Matter component (NFW-like or Burkert profile, your code uses exponential-like form)
    v_dm_sq = np.zeros_like(r_m, dtype=float)
    rs_dm_m = rs_dm_kpc_val * KPC_TO_M
    if np.any(non_zero_r_mask) and rs_dm_m != 0 : # Check rs_dm_m to avoid division by zero in exp
        r_m_nz = r_m[non_zero_r_mask]
        exp_term_dm_nz = np.exp(-r_m_nz / rs_dm_m)
        v_dm_sq[non_zero_r_mask] = (G * M_dm_val * M_SUN * (1 - exp_term_dm_nz)) / r_m_nz # Your original form

    v_total_sq = v_baryon_sq + v_dm_sq
    v_total_sq[v_total_sq < 0] = 0
    return np.sqrt(v_total_sq) * 1e-3

def rotation_curve_MOND_ext(r_kpc, M_solar_val, rs_kpc_baryon_val, g_ext_val):
    """
    MOND Model with external field effect:
    Units: r_kpc, rs_kpc_baryon in kpc; M_solar in M_sun; g_ext in m/s^2. Returns km/s.
    """
    r_m = np.asarray(r_kpc) * KPC_TO_M
    
    v_N_sq = np.zeros_like(r_m, dtype=float)
    a_N = np.zeros_like(r_m, dtype=float)
    non_zero_r_mask = r_m != 0

    if np.any(non_zero_r_mask):
        r_m_nz = r_m[non_zero_r_mask]
        # Baryonic potential (point mass for Newtonian acceleration a_N)
        v_N_sq[non_zero_r_mask] = (G * M_solar_val * M_SUN) / r_m_nz
        a_N[non_zero_r_mask] = v_N_sq[non_zero_r_mask] / r_m_nz

    # MOND interpolation function mu(x) = x / sqrt(1+x^2) is common,
    # your mu = a_N / (a0 + g_ext) seems different or a specific choice.
    # Let's use your formulation: mu = a_N / (A0_MOND + g_ext_val)
    # Ensure denominator is not zero. g_ext_val is usually positive or zero. A0_MOND is positive.
    denominator_mu = A0_MOND + g_ext_val
    if denominator_mu <= 0: # Should not happen with typical g_ext values
        mu = np.full_like(a_N, np.inf) # Or handle as error
    else:
        mu = a_N / denominator_mu
    
    # Your MOND velocity: v_MOND = v_N * np.sqrt(0.5 + 0.5 * np.sqrt(1 + 4 / mu**2))
    # Handle mu = 0 (where a_N = 0 or denominator_mu is huge)
    sqrt_inner_term = np.zeros_like(mu, dtype=float)
    non_zero_mu_mask = mu != 0
    if np.any(non_zero_mu_mask):
        sqrt_inner_term[non_zero_mu_mask] = np.sqrt(1 + 4 / mu[non_zero_mu_mask]**2)
    
    # For mu -> 0, sqrt(1+4/mu^2) -> 2/|mu|. If mu is always positive, -> 2/mu.
    # If a_N is 0, mu is 0. Then v_N is 0. So v_MOND should be 0.
    # Let's re-evaluate for mu=0 case.
    # If mu = 0, then 4/mu^2 -> inf. np.sqrt(inf) -> inf.
    # This implies v_MOND becomes complex or inf if v_N != 0 and mu=0, which seems problematic.
    # Standard simple MOND mu(x) = x / (1+x) or x/sqrt(1+x^2) where x = a_N/a0.
    # Your formula for v_MOND needs careful check for edge cases like mu=0.
    # For now, sticking to your formula and adding safeguards:

    v_mond_factor_sq = np.zeros_like(mu, dtype=float)
    valid_mu_mask = (mu != 0) # Avoid division by zero and ensure sqrt is real
    if np.any(valid_mu_mask):
      term_inside_outer_sqrt = 0.5 + 0.5 * sqrt_inner_term[valid_mu_mask]
      term_inside_outer_sqrt[term_inside_outer_sqrt < 0] = 0 # Ensure non-negative
      v_mond_factor_sq[valid_mu_mask] = term_inside_outer_sqrt
    
    v_MOND_sq = v_N_sq * v_mond_factor_sq
    v_MOND_sq[v_MOND_sq < 0] = 0

    return np.sqrt(v_MOND_sq) * 1e-3