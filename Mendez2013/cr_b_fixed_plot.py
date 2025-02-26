import cr_bound_b_fixed as crbf
import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(10, 6))
for fwhm, cr_bounds in crbf.results.items():
    plt.plot(crbf.delta_x_values, cr_bounds * 1e3, linestyle=crbf.line_styles[fwhm], color=crbf.line_colors[fwhm], label=f'FWHM = {fwhm}"')

plt.xlabel(r"Pixel size $\Delta x$ (arcsec)", fontsize=12)
plt.ylabel(r"Cramér-Rao Bound $\sigma_{CR}$ (mas)", fontsize=12)
plt.ylim(0, 60)
plt.title(r"Cramér-Rao Bound vs Pixel Size (fixed $B$ value)", fontsize=14)
plt.legend()
plt.grid()

# Save the figure
plt.savefig("cramer_rao_bound_vs_pixel_size_b_fixed.png", dpi=300)
plt.show()