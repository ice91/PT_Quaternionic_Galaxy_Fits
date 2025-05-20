# src/constants.py

"""
Physical and astronomical constants used in the analysis.
"""

G = 6.674e-11         # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30      # Solar mass (kg)
KPC_TO_M = 3.085e19   # 1 kpc to m (meters)
A0_MOND = 1.2e-10     # MOND acceleration constant (m/s^2)

# Default MCMC parameters (can be overridden by CLI arguments)
DEFAULT_NWALKERS = 32
DEFAULT_NSTEPS = 2000
DEFAULT_DISCARD = 500

# Default model parameters / initial guesses (can be refined)
INITIAL_RS_KPC = 3.0 # kpc
INITIAL_EPSILON_QUAT = 2.2 # Based on paper's findings for global fit
INITIAL_G_EXT_MOND = 1e-11 # m/s^2, example value