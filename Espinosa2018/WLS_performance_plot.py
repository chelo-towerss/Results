import numpy as np
import matplotlib.pyplot as plt
import functions as fn2

# Parameters
G = 2                                               # Gain of the detector (e^- / ADU)    
fs = 2000                                           # Sky background (ADU/arcsec)
RON = 5                                             # Read-out noise (e^-)
D = 0                                               # Dark current (e^-/s)
x_c = 0                                             # True centroid position (arsec)
u_plus = 1.164                                      # Corresponding u_plus for P = 0.9 (change to 1.525 for P = 0.97)
delta_x = 0.15                                      # Pixel size (arcsec)
n_pixels = 201                                      # Number of pixels
fwhm = 1.88                                         # FWHM (arcsec)
F_LOW_SNR = 840                                     # Total flux value for LOW_SNR (ADU)
F_HIGH_SNR = 13600                                  # Total flux value for HIGH_SNR (ADU)
n_simulations = 250                                 # Number of simulations per configuration
n_repetitions = 100                                 # Number of repetitions to obtain the average empirical variance

# Define the range of delta_x and flow values
delta_x_values = np.linspace(0.05, 0.6, 30)
delta_x_special_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  
Fluxes = np.array([840, 1612, 10002, 30080]) 

# Create the figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
fig.suptitle('Performance of the Weighted Least Squares Estimators (uniform weights)', fontsize=16, fontweight='bold')

# Precompute
precomputed_variances = {}
empirical_variance_results = {dx: np.zeros(n_repetitions) for dx in delta_x_special_values}

# calculate the variance of the WLS estimates for each delta x and each flow
for i, F in enumerate(Fluxes):
    variances = np.zeros_like(delta_x_values)
    var_LS_values = np.zeros_like(delta_x_values)
    var_CR_values = np.zeros_like(delta_x_values)
    empirical_means = np.zeros_like(delta_x_special_values)  

    # Standard calculation for each delta_x
    for j, delta_x in enumerate(delta_x_values):
        key = (F, delta_x)

        if key not in precomputed_variances:
            # Block simulation and estimation
            observations = fn2.simulate_observations(x_c, F, fs, n_simulations, n_pixels, delta_x, D, RON, G, fwhm)
            estimated_x_c = fn2.estimate_x_c_for_simulations_WLS(
                observations, "uniform", F, fs, delta_x, n_pixels, D, RON, G, fwhm, initial_guess=0.0
            )
            variance = np.var(estimated_x_c)

            # Calculate the Crámer-Rao Bound and the theoretical WLS variance
            var_LS, _, var_CR = fn2.calculate_variances(x_c, F, delta_x, n_pixels, fwhm, fs, D, RON, G)

            precomputed_variances[key] = (variance, var_LS, var_CR)
        else:
            variance, var_LS, var_CR = precomputed_variances[key]

        # Storing values
        variances[j] = variance
        var_LS_values[j] = var_LS
        var_CR_values[j] = var_CR

    # Extended calculation only for specific delta_x values
    for k, delta_x in enumerate(delta_x_special_values):
        for r in range(n_repetitions):
            observations = fn2.simulate_observations(x_c, F, fs, n_simulations, n_pixels, delta_x, D, RON, G, fwhm)
            estimated_x_c = fn2.estimate_x_c_for_simulations_WLS(
                observations, "uniform", F, fs, delta_x, n_pixels, D, RON, G, fwhm, initial_guess=0.0
            )
            empirical_variance_results[delta_x][r] = np.var(estimated_x_c)

        # Calculate average variances for this delta_x
        empirical_means[k] = np.mean(empirical_variance_results[delta_x])

    # Plot WLS variance and Cramér-Rao Bound
    axes[i].scatter(delta_x_values, np.sqrt(variances) * 1e3, marker='+', color='b', label='WLS Variance')
    axes[i].plot(delta_x_values, np.sqrt(var_LS_values) * 1e3, linestyle='-', color='blue', label=r'$\sigma_{WLS}$')
    axes[i].plot(delta_x_values, np.sqrt(var_CR_values) * 1e3, '--', color='red', label=r'$\sigma_{CR}$')

    # Plot averaged empirical variances
    axes[i].scatter(delta_x_special_values, np.sqrt(empirical_means) * 1e3, marker='o', color='green', label=r'Empirical Mean $\sqrt{\text{Var}(\tilde{\tau}_{WLS})}$')

    axes[i].set_xlabel(r'$\Delta x$ [arcsec]')
    axes[i].set_ylabel(r'$\sqrt{Var}$ [mas]')
    axes[i].set_title(f'Flux = {F}')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
