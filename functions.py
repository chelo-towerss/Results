import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import erf
from scipy import optimize
import scipy.integrate as integrate

# Gaussian sigma conversion
sigma_conversion = 1 / (2 * np.sqrt(2 * np.log(2)))

################# MODEL FUNCTIONS #################

def gaussian_psf(x, xc, sigma):
    """Gaussian PSF definition (vectorized approach)."""
    factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    return factor * np.exp(-(x - xc) ** 2 / (2 * sigma ** 2))

def g_function(x_c, fwhm, delta_x, n_pix):
    """Cuantization g_i(x_c) function using integration."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))

    sigma = fwhm * sigma_conversion

    # Compute the PSF integral for each pixel
    result = []
    for xi in x:
        x_plus = xi + delta_x / 2
        x_minus = xi - delta_x / 2

        integral_result, _ = integrate.quad(
            lambda k: gaussian_psf(k, x_c, sigma),
            x_minus, 
            x_plus,   
        )
        result.append(integral_result)

    return np.array(result)

def g_function_erf(x_c, fwhm, delta_x, n_pix):
    """Vectorized computation of g_i(x_c) using the error function (erf) for integration."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))

    # Sigma value
    sigma = fwhm * sigma_conversion

    # Pixel edged
    x_plus = (x + delta_x / 2 - x_c) / (np.sqrt(2) * sigma)
    x_minus = (x - delta_x / 2 - x_c) / (np.sqrt(2) * sigma)

    return 0.5 * (erf(x_plus) - erf(x_minus))

def g_function_norm(x_c, fwhm, delta_x, n_pix):
    """Cuantization g_i(x_c) function using scipy.stats.norm.cdf."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))  # Puntos centrales de los píxeles

    # Sigma value
    sigma = fwhm * sigma_conversion

    # Pixel edges
    x_minus = x - delta_x / 2  
    x_plus = x + delta_x / 2   

    # Use scipy.stats.norm.cdf to compute the integral
    result = norm.cdf(x_plus, loc=x_c, scale=sigma) - norm.cdf(x_minus, loc=x_c, scale=sigma)

    return result

def first_derivative_g_function(x_c, delta_x, n_pix, fwhm):
    """First derivative of g_i(x_c) function."""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))

    # Sigma value
    sigma = fwhm * sigma_conversion

    # Pixel edges
    x_plus = (x + delta_x / 2 - x_c)
    x_minus = (x - delta_x / 2 - x_c)

    factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    term1 = np.exp(-x_minus ** 2 / (2 * sigma ** 2))
    term2 = np.exp(-x_plus ** 2 / (2 * sigma ** 2))

    return factor * (term1 - term2)

def second_derivative_g_function(x_c, delta_x, n_pix, fwhm):
    """Second derivative of g_i(x_c) function"""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))

    # Pixel edges
    x_plus = (x + delta_x / 2 - x_c)
    x_minus = (x - delta_x / 2 - x_c)

    # Sigma value
    sigma = fwhm * sigma_conversion

    factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    term1 = np.exp(- x_minus ** 2 / (2 * sigma ** 2)) * (x_minus / sigma ** 2)
    term2 = np.exp(- x_plus ** 2 / (2 * sigma ** 2)) * (x_plus / sigma ** 2)

    return factor * (term1 - term2)

def third_derivative_g_function(x_c, delta_x, n_pix, fwhm):
    """Third derivative of g_i(x_c) function"""
    # Define spatial grid based on pixel size
    detector_extent = (n_pix - 1) * delta_x / 2  # Half of the detector extent
    x = np.linspace(-detector_extent, detector_extent, int(n_pix))

    # Pixel edges
    x_plus = (x + delta_x / 2 - x_c)
    x_minus = (x - delta_x / 2 - x_c)

    # Sigma value
    sigma = fwhm * sigma_conversion

    factor = 1 / (np.sqrt(2 * np.pi) * sigma ** 3)

    term1 = np.exp(- x_minus ** 2 / (2 * sigma ** 2)) * (x_minus ** 2 / (sigma ** 2) - 1)
    term2 = np.exp(- x_plus ** 2 / (2 * sigma ** 2)) * (x_plus ** 2 / (sigma ** 2) - 1)

    return factor(term1 - term2)

def background_B(fs, delta_x, D, RON, G):
    """Background per pixel as a function of pixel size [ADU/pix]."""
    return fs * delta_x + (D + RON**2) / G

def lambda_function(x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm):
    """Lambda function vectorized."""
    g_i = g_function_erf(x_c, fwhm, delta_x, n_pix)
    B_i = background_B(fs, delta_x, D, RON, G)
    return G * (F * g_i + B_i)

################ SNR's FUNCTIONS ################

def probability_integral(x):
    """Compute the probability integral."""
    integral, _ = integrate.quad(
        lambda k: np.exp(-k**2), 0, x)
    return integral

def P_function(u):
    """P(u) function computed using the error function (erf)."""
    return erf(u)

def signal_function(u, F, G):
    """"Signal function definition."""
    P = P_function(u)
    return G * F * P

def N_pix(u, delta_x, fwhm):
    """N_pix function definition."""
    return u / np.sqrt(np.log(2)) * fwhm / delta_x

def noise_function(u, F, fs, delta_x, RON, G, fwhm):
    """Noise function definition."""
    S = signal_function(u, F, G)
    Npix = N_pix(u, delta_x, fwhm)

    return np.sqrt(S + Npix * (G * fs * delta_x + RON ** 2))

def SNR(u, F, fs, delta_x, RON, G, fwhm):
    """Signal-to-noise ratio function definition."""
    S = signal_function(u, F, G)
    N = noise_function(u, F, fs, delta_x, RON, G, fwhm)
    return S / N

################# ESTIMATORS FUNCTIONS #################

# MAXIMUM LIKELIHOOD ESTIMATOR
def log_likelihood_ML(I, x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm):
    """Vectorized Log-likelihood function."""
    lambda_i = lambda_function(x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    return np.sum(I * np.log(lambda_i))

def estimate_x_c_numerical_ML(I, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess):
    """Numerical ML estimator."""
    objective = lambda x_c: -log_likelihood_ML(I, x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    bounds = [(None, None)]

    result = optimize.minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    return result.x[0]

def estimate_x_c_for_simulations_ML(data, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess):
    """Parallelized ML estimation for multiple simulations using vectorized processing."""
    return np.array([estimate_x_c_numerical_ML(I, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess) for I in data])

# WEIGHTED LEAST SQUARES ESTIMATOR
def likelihood_WLS(I, weight_type, x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm):
    """Weighted least squares function with different weighting strategies."""
    lambda_i = lambda_function(x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    K = 1  # Proportionality constant
    w_i = K / lambda_i if weight_type == "obs" else np.ones_like(I)  # w_i = K / lambda_i or 1
    return np.sum(w_i * (I - lambda_i) ** 2)

def estimate_x_c_numerical_WLS(I, weight_type, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess):
    """Numerical WLS estimator."""
    objective = lambda x_c: likelihood_WLS(I, weight_type, x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    bounds = [(None, None)]
    
    result = optimize.minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    return result.x[0]

def estimate_x_c_for_simulations_WLS(data, weight_type, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess):
    """Parallelized WLS estimation for multiple simulations using vectorized processing."""
    return np.array([estimate_x_c_numerical_WLS(I, weight_type, F, fs, delta_x, n_pix, D, RON, G, fwhm, initial_guess) for I in data])

# BIAS AND VARIANCE
def empirical_bias_and_variance(estimates, true_value, n_sum):
    """Calculate empirical bias and variance"""
    bias = 1 / n_sum * np.sum(true_value - estimates)
    variance = 1 / n_sum * np.sum((estimates - 1 / n_sum * np.sum(estimates))**2)
    return bias, variance

################# VARIANCES FUNCTIONS #################

def variance_CR(x_c, F, delta_x, n_pix, fwhm, fs, D, RON, G):
    """Compute the Cramér-Rao variance."""
    dg_i = first_derivative_g_function(x_c, delta_x, n_pix, fwhm)
    lambda_i = lambda_function(x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)

    denominator = np.sum((G * F * dg_i)**2 / lambda_i)

    return 1 / denominator
    
def variance_WLS(weight_type, x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm):
    """Compute the Weighted Least Squares variance."""
    lambda_i = lambda_function(x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    dlambda_i = G * F * first_derivative_g_function(x_c, delta_x, n_pix, fwhm)
    K = 1
    w_i = K / lambda_i if weight_type == "obs" else np.ones_like(n_pix)

    numerator = np.sum(w_i**2 * lambda_i * dlambda_i**2)
    denominator = np.sum(w_i * dlambda_i**2 )**2

    return numerator / denominator

def calculate_variances(x_c, F, delta_x, n_pix, fwhm, fs, D, RON, G):
    """Calcute the variances of the estimators and the Cramér-Rao bound."""
    
    var_LS = variance_WLS("uni", x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    var_WLS = variance_WLS("obs", x_c, F, fs, delta_x, n_pix, D, RON, G, fwhm)
    var_CR = variance_CR(x_c, F, delta_x, n_pix, fwhm, fs, D, RON, G)  # Equal to var_ML

    return var_LS, var_WLS, var_CR
    
############# SIMULATION FUNCTIONS #############

def simulate_observations(x_c, F, fs, n_simulations, n_pixels, delta_x, D, RON, G, fwhm):
    """Generate observation data for given fluxes with Poisson distribution."""
    
    lambda_values = lambda_function(x_c, F, fs, delta_x, n_pixels, D, RON, G, fwhm)
    simulations = np.random.poisson(lambda_values, size=(n_simulations, n_pixels))

    return np.array(simulations)

def generate_excel(filename, n_simulations, fluxes, x_c, fs, n_pixels, delta_x, D, RON, G, fwhm):
    """
    Generate an Excel file with simulations for each flux in the array.

    Parameters:
        filename: str
            Name of the output Excel file.
        n_simulations: int
            Number of simulations to generate.
        fluxes: list or array-like
            Array of flux values to generate simulations for.
        x_c, fs, n_pixels, delta_x, D, RON, G, fwhm:
            Parameters for the simulations.
    """
    with pd.ExcelWriter(filename) as writer:
        for F in fluxes:
            # Simulate observations for the current flux
            simulated_data = simulate_observations(x_c, F, fs, n_simulations, n_pixels, delta_x, D, RON, G, fwhm)
            
            # Convert NumPy array to DataFrame
            df = pd.DataFrame(simulated_data)
            
            # Write to Excel, naming the sheet by the flux value
            sheet_name = f"Flux_{int(F)}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
