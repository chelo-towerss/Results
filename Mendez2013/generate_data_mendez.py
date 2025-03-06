import numpy as np
import functions as fn

"""Script to generated data to reproduce the results from Mendez et al. 2013."""

np.random.seed(12)

# Parameters
G = 2                                               # Gain of the detector (e^- / ADU)    
fs = 2000                                           # Sky background (ADU/arcsec)
RON = 5                                             # Read-out noise (e^-)
D = 0                                               # Dark current (e^-/s)
TRUE_x_c = 0                                        # Centroid position (arcsec)
u_plus = 1.164                                      # Corresponding u_plus for P = 0.9 (change to 1.525 for P = 0.97)
delta_x = 0.15                                      # Pixel size (arcsec)
n_pix = 201                                         # Number of pixels
FWHM = 1.88                                         # FWHM (arcsec)
F_LOW_SNR = 740                                     # Total flux value (ADU) for low SNR
F_HIGH_SNR = 12050                                  # Total flux value (ADU) for high SNR
fluxes = np.array([F_LOW_SNR, F_HIGH_SNR])          # Flux values for low and high SNR

# SNR values
LOW_SNR_value = fn.SNR(u_plus, F_LOW_SNR, fs, delta_x, RON, G, FWHM)    # SNR = 12
HIGH_SNR_value = fn.SNR(u_plus, F_HIGH_SNR, fs, delta_x, RON, G, FWHM)  # SNR = 120

########## GENERATE DATA ##########
if __name__ == "__main__":
    filename = "Data_paper_1.xlsx"
    n_simulations = 250
    fn.generate_excel(filename, n_simulations, fluxes, TRUE_x_c, fs, n_pix, delta_x, D, RON, G, FWHM)

    print("LOW SNR value: ", LOW_SNR_value)
    print("HIGH SNR value: ", HIGH_SNR_value)
