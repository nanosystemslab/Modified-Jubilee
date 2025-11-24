import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------

csv_path = r"C:\Users\dvier\Documents\Thesis\Masters Thesis\Chapter 3 - Automated Dip Coating\3.3 Testing and Results\Data Analysis\dip_coating_results.csv"
df = pd.read_csv(csv_path)

# Extract speed labels from first row
speed_labels = df.iloc[0, 1:]
speeds = speed_labels.str.extract(r'([\d\.]+)\s*mm/s')[0].astype(float)

# Extract 5 raw measurements (rows 1–5)
sample_data = df.iloc[1:6, 1:].astype(float)

# Compute stats
avg_thickness = sample_data.mean(axis=0).values
std_thickness = sample_data.std(axis=0, ddof=1).values

# -------------------------------------------------------
# 2. Landau–Levich Model (t ∝ U^(2/3))
# -------------------------------------------------------

# Smooth curve for model
u_smooth = np.linspace(0.1, 12.0, 200)

# Scale model so it passes through your first data point (2.5 mm/s, ~7 µm)
u_ref = speeds.iloc[0]     # 2.5 mm/s
t_ref = avg_thickness[0]   # ~7.0 µm

k = t_ref / (u_ref ** (2.0 / 3.0))

t_smooth = k * u_smooth ** (2.0 / 3.0)

# -------------------------------------------------------
# 3. Plot Experimental Data with Error Bars + Model
# -------------------------------------------------------

plt.figure(figsize=(8, 6))

# Experimental data with error bars
plt.errorbar(
    speeds,
    avg_thickness,
    yerr=std_thickness,
    fmt="o",
    capsize=6,
    label="Experimental data",
    color="black"
)

# Landau–Levich smooth model curve
plt.plot(
    u_smooth,
    t_smooth,
    label="Landau–Levich model (t ∝ U$^{2/3}$)",
    linewidth=2
)

plt.xlabel("Withdrawal speed (mm/s)")
plt.ylabel("Film thickness (µm)")
plt.title("Experimental Film Thickness vs. Withdrawal Speed\nwith Landau–Levich Model Overlay")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("experiment_vs_landau_levich_plot.png", dpi=300)
plt.show()
