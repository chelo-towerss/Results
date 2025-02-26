import cr_bound_b_variable as crbv
import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(10, 6))
for flux, cr_bounds in crbv.results.items():
    plt.plot(crbv.delta_x_values, cr_bounds * 1e3, linestyle=crbv.line_styles[flux], label=f"Flux = {flux} ADU")

plt.xlabel(r"Pixel size $\Delta x$ (arcsec)", fontsize=12)
plt.ylabel(r"Cramér-Rao Bound $\sigma_{CR}$ (mas)", fontsize=12)
plt.ylim(0, 60)
plt.title("Cramér-Rao Bound vs Pixel Size", fontsize=14)

plt.legend(title="Flux Values", fontsize=10, title_fontsize=11, loc='upper left')
plt.grid(visible=True, linestyle='--', alpha=0.7)

plt.savefig("cramer_rao_bound_vs_pixel_size_b_variable.png", dpi=300)
plt.show()