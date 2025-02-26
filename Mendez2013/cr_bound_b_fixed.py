import numpy as np
import scipy.integrate as integrate

"""Script to compute CR bounds for different FWHM values and pixel sizes, using fixed B."""

# Parameters
G = 2                                        # Gain of the detector (e^- / ADU)    
B = 300                                      # Constant background per pixel (ADU/pix)
F_over_B = 10                                # Flux-background ratio (adim)
F = B * F_over_B                             # Total flux (ADU)
x_c = 0                                      # Centroid position (arcsec)
FWHM = np.array([0.5, 1.0, 1.5])             # FWHM values (arcsec)
n_pix = 201                                  # Number of pixels

# Gaussian sigma conversion
sigma_conversion = 1 / (2 * np.sqrt(2 * np.log(2)))  # Sigma values in arcsec

def gamma(x, x_c, sigma):
    """Definition of gamma"""
    return (x - x_c) ** 2 / (2 * sigma ** 2)

def compute_cr_bound(F, x_c, delta_x, FWHM):
    """Compute the Cramér-Rao bound as a function of pixel size."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, n_pix)

    # Sigma value
    sigma = FWHM * sigma_conversion

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
            lambda k: np.exp(-gamma(k, x_c, sigma)), xi_minus, xi_plus
        )

        denominator += numerator[i] / (1 + (F / (np.sqrt(2 * np.pi) * sigma * B)) * integral)

    # Cramér-Rao bound squared
    return 2 * np.pi * sigma**2 * B / (G * F**2) / denominator

# Compute CR bound for different FWHM values and pixel sizes
delta_x_values = np.linspace(0.01, 2.0, 500)  # Pixel size (arcsec)
results = {}

line_styles = {0.5: 'dotted', 1.0: 'solid', 1.5: 'dashed'}  # Line styles for different FWHM values
line_colors = {0.5: 'darkblue', 1.0: 'darkgoldenrod', 1.5: 'firebrick'}       # Line colors for different FWHM values

for fwhm in FWHM:
    sigma = fwhm * sigma_conversion
    cr_bounds = [compute_cr_bound(F, x_c, dx, fwhm) for dx in delta_x_values]
    results[fwhm] = np.sqrt(cr_bounds)  # Take the square root to get sigma_CR
