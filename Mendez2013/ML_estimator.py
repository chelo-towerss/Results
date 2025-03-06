import numpy as np
import functions as fn
import pandas as pd

"""ESTIMATION USING THE MAXIMUM LIKELIHOOD ESTIMATOR."""

np.random.seed(12)

# Parameters
G = 2                                               # Gain of the detector (e^- / ADU)    
fs = 2000                                           # Sky background (ADU/arcsec)
RON = 5                                             # Read-out noise (e^-)
D = 0                                               # Dark current (e^-/s)
TRUE_x_c = 0                                        # True centroid position (arsec)
u_plus = 1.164                                      # Corresponding u_plus for P = 0.9 (change to 1.525 for P = 0.97)
delta_x = 0.15                                      # Pixel size (arcsec)
n_pix = 201                                         # Number of pixels
FWHM = 1.88                                         # FWHM (arcsec)
F_LOW_SNR = 840                                     # Total flux value for LOW_SNR (ADU)
F_HIGH_SNR = 13600                                  # Total flux value for HIGH_SNR (ADU)

# Load data
generated_Data = 'Data_paper_Mendez.xlsx'
data_LOW_SNR = np.array(pd.read_excel(generated_Data, sheet_name='Flux_740'))
data_HIGH_SNR = np.array(pd.read_excel(generated_Data, sheet_name='Flux_12050'))

def main():
    # Set initial guess for the ML estimator
    INITIAL_GUESS = np.array([0.0])

    # Set number of simulations using the number of rows in the data
    n_sum = data_LOW_SNR.shape[0]  

    # Compute estimators for LOW_SNR
    estimated_x_c_LOW_SNR = fn.estimate_x_c_for_simulations_ML(data_LOW_SNR, F_LOW_SNR, fs, delta_x, n_pix, D, RON, G, FWHM, INITIAL_GUESS)
    mean_x_c_LOW_SNR = np.mean(estimated_x_c_LOW_SNR)
    bias_LOW, variance_LOW = fn.empirical_bias_and_variance(estimated_x_c_LOW_SNR, TRUE_x_c, n_sum)

    # Compute estimators for HIGH_SNR
    estimated_x_c_HIGH_SNR = fn.estimate_x_c_for_simulations_ML(data_HIGH_SNR, F_HIGH_SNR, fs, delta_x, n_pix, D, RON, G, FWHM, INITIAL_GUESS)
    mean_x_c_HIGH_SNR = np.mean(estimated_x_c_HIGH_SNR)
    bias_HIGH, variance_HIGH = fn.empirical_bias_and_variance(estimated_x_c_HIGH_SNR, TRUE_x_c, n_sum)

    # Compute Cramér-Rao bound for each scenario
    cramer_rao_LOW = fn.variance_CR(TRUE_x_c, F_LOW_SNR, delta_x, n_pix, FWHM, fs, D, RON, G)
    cramer_rao_HIGH = fn.variance_CR(TRUE_x_c, F_HIGH_SNR, delta_x, n_pix, FWHM, fs, D, RON, G)

    # Show results
    print(f"Results ML estimator")
    print(f"=====================")
    print(f"Low SNR:")
    print(f"  True x_c value                     (arsec): {TRUE_x_c}")
    print(f"  Mean estimated x_c                 (arsec): {mean_x_c_LOW_SNR}")
    print(f"  Empiric bias                       (arsec): {bias_LOW}")
    print(f"  Empiric variance                 (arsec^2): {variance_LOW}")
    print(f"  Cramér-Rao Bound (Var)           (arsec^2): {cramer_rao_LOW}")
    print(f"  Empiric standard deviation           (mas): {np.sqrt(variance_LOW) * 1e3}")
    print(f"  Cramér-Rao Bound (STD)               (mas): {np.sqrt(cramer_rao_LOW) * 1e3}")
    
    print(f"=====================")
    print(f"High SNR:")
    print(f"  True x_c value                     (arsec): {TRUE_x_c}")
    print(f"  Mean estimated x_c                 (arsec): {mean_x_c_HIGH_SNR}")
    print(f"  Empiric bias                       (arsec): {bias_HIGH}")
    print(f"  Empiric variance                 (arsec^2): {variance_HIGH}")
    print(f"  Cramér-Rao Bound (Var)           (arsec^2): {cramer_rao_HIGH}")
    print(f"  Empiric standard deviation           (mas): {np.sqrt(variance_HIGH) * 1e3}")
    print(f"  Cramér-Rao Bound (STD)               (mas): {np.sqrt(cramer_rao_HIGH) * 1e3}")
    

if __name__ == "__main__":
    main()
