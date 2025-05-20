# src/sparc_fitter_cli.py
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

# Import from local modules
from .constants import (DEFAULT_NWALKERS, DEFAULT_NSTEPS, DEFAULT_DISCARD,
                       INITIAL_RS_KPC, INITIAL_EPSILON_QUAT, INITIAL_G_EXT_MOND)
from .models import (rotation_curve_quat_simple, rotation_curve_LCDM,
                    rotation_curve_MOND_ext)
from .mcmc_utils import (log_probability_quat, log_probability_LCDM, log_probability_MOND,
                         get_mcmc_results, calculate_goodness_of_fit)

def run_sparc_analysis(data_filepath, output_dir, 
                       nwalkers=DEFAULT_NWALKERS, nsteps=DEFAULT_NSTEPS, 
                       discard=DEFAULT_DISCARD, plot_individual_fits=True):
    """
    Main function to run the SPARC galaxy fitting analysis.
    """
    start_time = time.time()
    print(f"--- Starting SPARC Galaxy Rotation Curve Analysis ---")
    print(f"Data file: {data_filepath}")
    print(f"Output directory: {output_dir}")
    print(f"MCMC parameters: walkers={nwalkers}, steps={nsteps}, discard={discard}")
    print(f"Plot individual fits: {plot_individual_fits}")
    print("-" * 50)

    # --- Data Loading and Preprocessing ---
    try:
        # IMPORTANT: Adjust column names and processing based on the actual SPARC file format
        # This is a placeholder based on common SPARC structures
        # Assuming space-separated, header might be present or not.
        # Best to inspect your specific sparc_data_all.txt or similar.
        raw_data = pd.read_csv(data_filepath, sep=r'\s+', comment='#', header=0, na_filter=False) # header=0 if first line is header
        
        # Example column renaming (ADAPT TO YOUR FILE!)
        column_map = {
            'GalaxyName': 'galaxy_id', # Or 'Name'
            'Rad': 'rad[kpc]',        # Or 'R(kpc)'
            'Vobs': 'vobs[km/s]',     # Or 'Vc(km/s)'
            'e_Vobs': 'errv[km/s]',   # Or 'errVc(km/s)'
            'Mstar': 'mass[M_sun]'    # If SPARC file has a total stellar mass column directly (rare)
                                      # Otherwise, calculate from Mdisk, Mbulge
            # Add Mdisk, Mbulge if you need to calculate total stellar mass
            # 'Mdisk': 'disk_mass_1e9Msun',
            # 'Mbulge': 'bulge_mass_1e9Msun',
        }
        # Filter for necessary columns and rename
        available_cols_in_map = {k: v for k, v in column_map.items() if k in raw_data.columns}
        processed_data = raw_data[list(available_cols_in_map.keys())].rename(columns=available_cols_in_map)

        # Calculate total stellar mass if not directly available
        # EXAMPLE: if 'disk_mass_1e9Msun' in processed_data.columns:
        #     processed_data['mass[M_sun]'] = processed_data['disk_mass_1e9Msun'] * 1e9
        #     if 'bulge_mass_1e9Msun' in processed_data.columns:
        #         processed_data['mass[M_sun]'] += processed_data['bulge_mass_1e9Msun'].fillna(0) * 1e9
        # else:
        #     raise ValueError("Stellar mass columns (e.g., Mdisk) not found or mapped correctly.")

        # Ensure 'mass[M_sun]' column exists after processing
        if 'mass[M_sun]' not in processed_data.columns:
             print("WARNING: 'mass[M_sun]' column not found after attempting to map/calculate. Using a placeholder if needed for MCMC initial guess, but this is not ideal.")
             # This part needs robust handling based on your data.

    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{data_filepath}'. Please provide a valid path.")
        return
    except Exception as e:
        print(f"ERROR loading or processing data file: {e}")
        print("Please ensure the SPARC data file is correctly formatted and column names are mapped appropriately in the script.")
        return

    galaxy_ids = processed_data['galaxy_id'].unique()
    all_results_list = []

    for gal_idx, galaxy_name in enumerate(galaxy_ids):
        print(f"\n--- Processing Galaxy {gal_idx+1}/{len(galaxy_ids)}: {galaxy_name} ---")
        
        galaxy_data_df = processed_data[processed_data['galaxy_id'] == galaxy_name].copy()
        
        # Data cleaning for the current galaxy
        galaxy_data_df.dropna(subset=['rad[kpc]', 'vobs[km/s]', 'errv[km/s]', 'mass[M_sun]'], inplace=True)
        galaxy_data_df = galaxy_data_df[
            (galaxy_data_df['vobs[km/s]'] > 0) & 
            (galaxy_data_df['errv[km/s]'] > 0) # Errors should be positive
        ]
        # Ensure at least N_params + 1 data points for a meaningful fit
        if len(galaxy_data_df) < 5: # e.g., for 4-param LCDM
            print(f"Skipping {galaxy_name}: Insufficient valid data points ({len(galaxy_data_df)}).")
            continue

        r_obs_kpc = galaxy_data_df['rad[kpc]'].values
        v_obs_kms = galaxy_data_df['vobs[km/s]'].values
        err_v_kms = galaxy_data_df['errv[km/s]'].values.clip(min=1e-3) # Clip to small positive to avoid division by zero

        # Use the stellar mass of the galaxy for MCMC initial guess
        # Assuming 'mass[M_sun]' is now correctly populated for each galaxy group
        baryonic_mass_solar = galaxy_data_df['mass[M_sun]'].iloc[0] # Should be consistent for the galaxy
        if pd.isna(baryonic_mass_solar) or baryonic_mass_solar <= 0:
            print(f"Warning: Invalid baryonic mass ({baryonic_mass_solar}) for {galaxy_name}. Using a default guess.")
            baryonic_mass_solar = 1e10 # Fallback default

        current_galaxy_results = {'Galaxy': galaxy_name}

        # --- 1. Quaternionic Model Fit ---
        print("Fitting Quaternionic Model...")
        ndim_quat = 3
        # Initial guess slightly perturbed around sensible values
        p0_quat_center = np.array([baryonic_mass_solar, INITIAL_RS_KPC, INITIAL_EPSILON_QUAT])
        p0_quat = p0_quat_center + 1e-2 * p0_quat_center * np.random.randn(nwalkers, ndim_quat)
        # Ensure initial positions are within prior bounds
        for i in range(nwalkers):
            while not np.isfinite(log_prior_quat(p0_quat[i,:])):
                 p0_quat[i,:] = p0_quat_center + 1e-1 * p0_quat_center * np.random.randn(ndim_quat)


        sampler_quat = emcee.EnsembleSampler(nwalkers, ndim_quat, log_probability_quat, 
                                             args=(r_obs_kpc, v_obs_kms, err_v_kms, rotation_curve_quat_simple))
        sampler_quat.run_mcmc(p0_quat, nsteps, progress=True, store=True)
        
        params_quat, samples_quat, ci_quat = get_mcmc_results(sampler_quat, discard, ndim_quat)
        M_s_q, rs_q, eps_q = params_quat
        v_fit_quat = rotation_curve_quat_simple(r_obs_kpc, M_s_q, rs_q, eps_q)
        chi2_q, rchi2_q, aic_q, bic_q = calculate_goodness_of_fit(v_obs_kms, v_fit_quat, err_v_kms, ndim_quat)
        
        current_galaxy_results.update({
            'M_solar_Quat': M_s_q, 'rs_kpc_Quat': rs_q, 'epsilon_Quat': eps_q,
            'chi2_Quat': chi2_q, 'red_chi2_Quat': rchi2_q, 'AIC_Quat': aic_q, 'BIC_Quat': bic_q,
            # Add CIs if needed: 'M_solar_Quat_low': ci_quat[0]['low_1sigma'], ...
        })

        # --- 2. LCDM Model Fit ---
        print("Fitting LCDM Model...")
        ndim_lcdm = 4
        # Initial guess for LCDM
        p0_lcdm_center = np.array([baryonic_mass_solar, INITIAL_RS_KPC, baryonic_mass_solar * 5, 15.0]) # M_dm, rs_dm
        p0_lcdm = p0_lcdm_center + 1e-2 * p0_lcdm_center * np.random.randn(nwalkers, ndim_lcdm)
        for i in range(nwalkers): # Ensure priors
            while not np.isfinite(log_prior_LCDM(p0_lcdm[i,:])):
                 p0_lcdm[i,:] = p0_lcdm_center + 1e-1 * p0_lcdm_center * np.random.randn(ndim_lcdm)

        sampler_lcdm = emcee.EnsembleSampler(nwalkers, ndim_lcdm, log_probability_LCDM,
                                             args=(r_obs_kpc, v_obs_kms, err_v_kms, rotation_curve_LCDM))
        sampler_lcdm.run_mcmc(p0_lcdm, nsteps, progress=True, store=True)

        params_lcdm, samples_lcdm, ci_lcdm = get_mcmc_results(sampler_lcdm, discard, ndim_lcdm)
        M_s_l, rs_b_l, M_dm_l, rs_dm_l = params_lcdm
        v_fit_lcdm = rotation_curve_LCDM(r_obs_kpc, M_s_l, rs_b_l, M_dm_l, rs_dm_l)
        chi2_l, rchi2_l, aic_l, bic_l = calculate_goodness_of_fit(v_obs_kms, v_fit_lcdm, err_v_kms, ndim_lcdm)

        current_galaxy_results.update({
            'M_solar_LCDM': M_s_l, 'rs_kpc_baryon_LCDM': rs_b_l, 
            'M_dm_LCDM': M_dm_l, 'rs_dm_kpc_LCDM': rs_dm_l,
            'chi2_LCDM': chi2_l, 'red_chi2_LCDM': rchi2_l, 'AIC_LCDM': aic_l, 'BIC_LCDM': bic_l,
        })

        # --- 3. MOND Model Fit ---
        print("Fitting MOND Model...")
        ndim_mond = 3
        p0_mond_center = np.array([baryonic_mass_solar, INITIAL_RS_KPC, INITIAL_G_EXT_MOND])
        p0_mond = p0_mond_center + 1e-2 * p0_mond_center * np.random.randn(nwalkers, ndim_mond)
        # Ensure g_ext is positive for initial guess if center is 0 and random makes it negative
        p0_mond[:, 2] = np.abs(p0_mond[:, 2]) if INITIAL_G_EXT_MOND == 0 else p0_mond[:, 2]
        for i in range(nwalkers): # Ensure priors
            while not np.isfinite(log_prior_MOND(p0_mond[i,:])):
                 p0_mond[i,:] = p0_mond_center + 1e-1 * p0_mond_center * np.random.randn(ndim_mond)
                 p0_mond[i, 2] = np.abs(p0_mond[i, 2]) if INITIAL_G_EXT_MOND == 0 else p0_mond[i, 2]


        sampler_mond = emcee.EnsembleSampler(nwalkers, ndim_mond, log_probability_MOND,
                                             args=(r_obs_kpc, v_obs_kms, err_v_kms, rotation_curve_MOND_ext))
        sampler_mond.run_mcmc(p0_mond, nsteps, progress=True, store=True)

        params_mond, samples_mond, ci_mond = get_mcmc_results(sampler_mond, discard, ndim_mond)
        M_s_m, rs_b_m, g_ext_m = params_mond
        v_fit_mond = rotation_curve_MOND_ext(r_obs_kpc, M_s_m, rs_b_m, g_ext_m)
        chi2_m, rchi2_m, aic_m, bic_m = calculate_goodness_of_fit(v_obs_kms, v_fit_mond, err_v_kms, ndim_mond)

        current_galaxy_results.update({
            'M_solar_MOND': M_s_m, 'rs_kpc_baryon_MOND': rs_b_m, 'g_ext_MOND': g_ext_m,
            'chi2_MOND': chi2_m, 'red_chi2_MOND': rchi2_m, 'AIC_MOND': aic_m, 'BIC_MOND': bic_m,
        })
        
        # --- Determine Best Model based on AIC ---
        aics = {
            'Quaternionic': aic_q,
            'LCDM': aic_l,
            'MOND': aic_m
        }
        # Filter out NaN AICs before finding the minimum
        valid_aics = {k: v for k, v in aics.items() if pd.notna(v)}
        if valid_aics:
            best_model_name = min(valid_aics, key=valid_aics.get)
        else:
            best_model_name = "N/A (all AICs NaN)"
        current_galaxy_results['Best_Model_AIC'] = best_model_name
        
        all_results_list.append(current_galaxy_results)

        # --- Plotting for the current galaxy ---
        if plot_individual_fits:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(r_obs_kpc, v_obs_kms, yerr=err_v_kms, fmt='o', color='black', 
                        label=f'SPARC Data ({galaxy_name})', alpha=0.7, capsize=3, markersize=5)
            
            ax.plot(r_obs_kpc, v_fit_quat, color='blue', linestyle='-', linewidth=2,
                    label=f'Quaternionic (AIC: {aic_q:.1f})\n  $M_s={M_s_q:.1e}, r_s={rs_q:.1f}, \\epsilon={eps_q:.2f}$')
            ax.plot(r_obs_kpc, v_fit_lcdm, color='red', linestyle='--', linewidth=2,
                    label=f'ΛCDM (AIC: {aic_l:.1f})\n  $M_s={M_s_l:.1e}, M_d={M_dm_l:.1e}$')
            ax.plot(r_obs_kpc, v_fit_mond, color='green', linestyle=':', linewidth=2,
                    label=f'MOND (AIC: {aic_m:.1f})\n  $M_s={M_s_m:.1e}, g_{{ext}}={g_ext_m:.1e}$')

            ax.set_xlabel('Radius (kpc)', fontsize=12)
            ax.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
            ax.set_title(f'Rotation Curve Fits for {galaxy_name}', fontsize=14)
            ax.legend(fontsize=9, loc='lower right') # Adjust legend location as needed
            ax.set_xscale('log') # Common for rotation curves
            ax.grid(True, which="both", ls="-", alpha=0.5)
            plt.tight_layout()
            
            plot_filename = os.path.join(output_dir, f"{galaxy_name.replace(' ', '_')}_fit_comparison.png")
            try:
                fig.savefig(plot_filename, dpi=150)
                print(f"Saved plot: {plot_filename}")
            except Exception as e_plot:
                print(f"ERROR saving plot for {galaxy_name}: {e_plot}")
            plt.close(fig)

    # --- Aggregate and Save Results ---
    if not all_results_list:
        print("No galaxies were successfully processed and fitted.")
    else:
        df_final_results = pd.DataFrame(all_results_list)
        csv_filename = os.path.join(output_dir, "sparc_fit_comparison_summary.csv")
        try:
            df_final_results.to_csv(csv_filename, index=False, float_format='%.3e')
            print(f"\nFull results saved to: {csv_filename}")
        except Exception as e_csv:
            print(f"ERROR saving CSV results: {e_csv}")

        # --- Print Overall Statistics ---
        print("\n--- Overall Goodness-of-Fit (Summed χ² / Summed DoF) ---")
        for model_prefix in ['Quat', 'LCDM', 'MOND']:
            total_chi2 = df_final_results[f'chi2_{model_prefix}'].sum()
            n_params = ndim_quat if model_prefix == 'Quat' else (ndim_lcdm if model_prefix == 'LCDM' else ndim_mond)
            
            total_dof_model = 0
            for gal_name_iter in df_final_results['Galaxy']: # Iterate through successfully fitted galaxies
                 gal_data_iter_df = processed_data[processed_data['galaxy_id'] == gal_name_iter].copy()
                 gal_data_iter_df.dropna(subset=['rad[kpc]', 'vobs[km/s]', 'errv[km/s]', 'mass[M_sun]'], inplace=True)
                 gal_data_iter_df = gal_data_iter_df[
                    (gal_data_iter_df['vobs[km/s]'] > 0) & 
                    (gal_data_iter_df['errv[km/s]'] > 0) 
                 ]
                 total_dof_model += (len(gal_data_iter_df) - n_params)
            
            if total_dof_model > 0:
                total_reduced_chi2 = total_chi2 / total_dof_model
                print(f"Total Reduced Chi-squared ({model_prefix}): {total_reduced_chi2:.2f} (χ²={total_chi2:.1f}, DoF={total_dof_model})")
            else:
                print(f"Total Reduced Chi-squared ({model_prefix}): N/A (DoF=0)")

        print("\n--- Best Model Counts (based on AIC) ---")
        print(df_final_results['Best_Model_AIC'].value_counts())
    
    end_time = time.time()
    print(f"\n--- Analysis finished in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit SPARC galaxy rotation curves with Quaternionic, LCDM, and MOND models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_filepath", 
        type=str, 
        help="Path to the SPARC data file (e.g., from http://astroweb.cwru.edu/SPARC/)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results", 
        help="Directory to save CSV results and plots."
    )
    parser.add_argument(
        "--nwalkers", 
        type=int, 
        default=DEFAULT_NWALKERS, 
        help="Number of MCMC walkers."
    )
    parser.add_argument(
        "--nsteps", 
        type=int, 
        default=DEFAULT_NSTEPS, 
        help="Number of MCMC steps per walker."
    )
    parser.add_argument(
        "--discard", 
        type=int, 
        default=DEFAULT_DISCARD, 
        help="Number of burn-in MCMC steps to discard."
    )
    parser.add_argument(
        "--no_plots", 
        action="store_true", 
        help="Disable saving of individual galaxy fit plots."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    run_sparc_analysis(
        data_filepath=args.data_filepath,
        output_dir=args.output_dir,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        discard=args.discard,
        plot_individual_fits=not args.no_plots
    )