import numpy as np
import scipy.integrate as integrate

"""Script to compute CR bounds for different FWHM values and pixel sizes, using variable B."""

# Parameters
G = 2                                               # Gain of the detector (e^- / ADU)
f_s = 2000                                          # Sky background (ADU/arcsec)
x_c = 0                                             # Centroid position (arcsec)
RON = 5                                             # Read-out noise (e^-)
D = 0                                               # Dark current (e^-/s)    
FWHM = 1.0                                          # FWHM (arcsec) (Gaussian source centered on a pixel)
F = np.array([1000, 2000, 5000, 10000, 50000])      # Total flux values (ADU)
n_pix = 201                                         # Number of pixels

# Gaussian sigma conversion
sigma_conversion = 1 / (2 * np.sqrt(2 * np.log(2)))  # Sigma values in arcsec

def background_B(f_s, delta_x, D, RON, G):
    """Background per pixel as a function of pixel size [ADU/pix]."""
    return f_s * delta_x + (D + RON**2) / G

def gamma(x, x_c, sigma):
    """Definition of gamma"""
    return (x - x_c) ** 2 / (2 * sigma ** 2)

def compute_cr_bound(F, x_c, f_s, delta_x, D, RON, G, fwhm):
    """Compute the Cramér-Rao bound as a function of pixel size."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Total extent of the detector
    x = np.linspace(-detector_extent, detector_extent, n_pix)

    # Sigma value
    sigma = fwhm * sigma_conversion

    # Calculate pixel edges
    x_minus = x - delta_x / 2
    x_plus = x + delta_x / 2

    # Numerator: (e^(-gamma(x_minus)) - e^(-gamma(x_plus)))^2
    exp_diff = np.exp(-gamma(x_minus, x_c, sigma)) - np.exp(-gamma(x_plus, x_c, sigma))
    numerator = exp_diff**2

    # Denominator: Integral and sum components
    denominator = 0
    for i in range(len(x)):
        xi_minus = x[i] - delta_x / 2
        xi_plus = x[i] + delta_x / 2

        # Integral of exp(-gamma(x))
        integral, _ = integrate.quad(
            lambda k: np.exp(-gamma(k, 0, sigma)), xi_minus, xi_plus
        )

        denominator += numerator[i] / (1 + (F / (np.sqrt(2 * np.pi) * sigma * background_B(f_s, delta_x, D, RON, G))) * integral)

    # Cramér-Rao bound squared
    return 2 * np.pi * sigma**2 * background_B(f_s, delta_x, D, RON, G) / (G * F**2) / denominator


# Different pixel sizes (arsec)
delta_x_values = np.linspace(0.01, 2.0, 500)

results = {}

line_styles = {1000: 'dashed', 2000: 'dashdot', 5000: 'solid', 10000: 'dotted', 50000: 'solid'}  # Line styles for different F values

# Compute CR bound for different F values and pixel sizes
for flux in F:
    cr_bounds = [compute_cr_bound(flux, x_c, f_s, dx, D, RON, G, FWHM) for dx in delta_x_values]
    results[flux] = np.sqrt(cr_bounds)  # Take the square root to get sigma_CR