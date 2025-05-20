# src/mcmc_utils.py
import numpy as np

# --- Quaternionic Model Priors & Likelihood ---
def log_likelihood_quat(theta, r_kpc_data, v_obs, v_err, model_func):
    M_solar_val, rs_kpc_val, epsilon_val = theta
    if M_solar_val <=0 or rs_kpc_val <=0: # Basic physical check
        return -np.inf
    v_model = model_func(r_kpc_data, M_solar_val, rs_kpc_val, epsilon_val)
    if np.any(np.isnan(v_model)): # Check for NaNs from model
        return -np.inf
    sigma2 = v_err**2
    return -0.5 * np.sum((v_obs - v_model)**2 / sigma2 + np.log(2 * np.pi * sigma2)) # Gaussian logL

def log_prior_quat(theta):
    M_solar_val, rs_kpc_val, epsilon_val = theta
    # Prior ranges from your original code, can be adjusted
    if 1e7 < M_solar_val < 1e12 and \
       0.05 < rs_kpc_val < 50 and \
       1e-3 < epsilon_val < 1e1:  # Adjusted epsilon range for physical sense?
        return 0.0 # Flat prior
    return -np.inf

def log_probability_quat(theta, r_kpc_data, v_obs, v_err, model_func):
    lp = log_prior_quat(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_quat(theta, r_kpc_data, v_obs, v_err, model_func)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- LCDM Model Priors & Likelihood ---
def log_likelihood_LCDM(theta, r_kpc_data, v_obs, v_err, model_func):
    M_solar_val, rs_kpc_baryon_val, M_dm_val, rs_dm_kpc_val = theta
    if M_solar_val <=0 or rs_kpc_baryon_val <=0 or M_dm_val < 0 or rs_dm_kpc_val <=0: # DM mass can be 0
        return -np.inf
    v_model = model_func(r_kpc_data, M_solar_val, rs_kpc_baryon_val, M_dm_val, rs_dm_kpc_val)
    if np.any(np.isnan(v_model)):
        return -np.inf
    sigma2 = v_err**2
    return -0.5 * np.sum((v_obs - v_model)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior_LCDM(theta):
    M_solar_val, rs_kpc_baryon_val, M_dm_val, rs_dm_kpc_val = theta
    if 1e7 < M_solar_val < 1e12 and \
       0.05 < rs_kpc_baryon_val < 50 and \
       0 < M_dm_val < 1e13 and \
       0.1 < rs_dm_kpc_val < 100:
        return 0.0
    return -np.inf

def log_probability_LCDM(theta, r_kpc_data, v_obs, v_err, model_func):
    lp = log_prior_LCDM(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_LCDM(theta, r_kpc_data, v_obs, v_err, model_func)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- MOND Model Priors & Likelihood ---
def log_likelihood_MOND(theta, r_kpc_data, v_obs, v_err, model_func):
    M_solar_val, rs_kpc_baryon_val, g_ext_val = theta
    if M_solar_val <=0 or rs_kpc_baryon_val <=0 or g_ext_val < 0:
        return -np.inf
    v_model = model_func(r_kpc_data, M_solar_val, rs_kpc_baryon_val, g_ext_val)
    if np.any(np.isnan(v_model)):
        return -np.inf
    sigma2 = v_err**2
    return -0.5 * np.sum((v_obs - v_model)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior_MOND(theta):
    M_solar_val, rs_kpc_baryon_val, g_ext_val = theta
    # g_ext usually very small, around 1e-11 to 1e-12 m/s^2
    if 1e7 < M_solar_val < 1e12 and \
       0.05 < rs_kpc_baryon_val < 50 and \
       0 <= g_ext_val < 1e-10: # g_ext can be zero
        return 0.0
    return -np.inf

def log_probability_MOND(theta, r_kpc_data, v_obs, v_err, model_func):
    lp = log_prior_MOND(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_MOND(theta, r_kpc_data, v_obs, v_err, model_func)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- General MCMC Helper ---
def get_mcmc_results(sampler, discard_steps, n_params):
    """Extracts median parameters and confidence intervals from emcee sampler."""
    samples = sampler.get_chain(discard=discard_steps, flat=True)
    median_params = np.median(samples, axis=0)
    
    param_cis = []
    for i in range(n_params):
        # 16th, 50th, 84th percentiles for median and 1-sigma error
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        param_cis.append({'median': p50, 'low_1sigma': p16, 'high_1sigma': p84})
        # Overwrite median_params with the 50th percentile for consistency
        median_params[i] = p50 
        
    return median_params, samples, param_cis

def calculate_goodness_of_fit(v_obs, v_model, v_err, n_params):
    """Calculates chi2, reduced_chi2, AIC, BIC."""
    N_data = len(v_obs)
    chi2 = np.sum(((v_obs - v_model) / v_err)**2)
    dof = N_data - n_params
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    AIC = chi2 + 2 * n_params
    BIC = chi2 + n_params * np.log(N_data) if N_data > 0 else np.nan # Avoid log(0)
    return chi2, red_chi2, AIC, BIC