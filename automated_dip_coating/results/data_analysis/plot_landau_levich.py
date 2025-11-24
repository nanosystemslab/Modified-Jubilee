import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# Landau–Levich model shape: t ∝ U^(2/3)
# -------------------------------------------

# Define a range of withdrawal speeds (mm/s)
# Avoid 0 to prevent division issues; start at a small positive value.
u = np.linspace(0.1, 12.0, 200)  # speeds from 0.1 to 12 mm/s

# Optional: choose a scaling constant so the curve
# roughly matches your first data point (~7 µm at 2.5 mm/s).
# t = k * U^(2/3)
u_ref = 2.5       # mm/s
t_ref = 7.0       # µm (your average at 2.5 mm/s)
k = t_ref / (u_ref ** (2.0 / 3.0))

# Compute theoretical thickness (in µm)
t_model = k * u ** (2.0 / 3.0)

# -------------------------------------------
# Plot
# -------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(u, t_model, label="Landau–Levich model (t ∝ U$^{2/3}$)")

# (Optional) highlight the discrete speeds you actually tested
speeds_measured = np.array([2.5, 5.0, 7.5, 10.0])
t_model_measured = k * speeds_measured ** (2.0 / 3.0)
plt.scatter(speeds_measured, t_model_measured, color="black", zorder=3,
            label="Model at measured speeds")

plt.xlabel("Withdrawal speed (mm/s)")
plt.ylabel("Film thickness (µm)")
plt.title("Landau–Levich Model: Film Thickness vs. Withdrawal Speed")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("landau_levich_plot.png", dpi=300)
plt.show()