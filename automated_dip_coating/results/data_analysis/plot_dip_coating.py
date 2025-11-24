import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load the data ---
# Use an r-string (raw string) to avoid errors with backslashes in Windows paths.
csv_path = r"C:\Users\dvier\Documents\Thesis\Masters Thesis\Chapter 3 - Automated Dip Coating\3.3 Testing and Results\Data Analysis\dip_coating_results.csv"

df = pd.read_csv(csv_path)

# Extract the column labels for the speeds from the first row (excluding the first column)
speed_labels = df.iloc[0, 1:]

# Parse withdrawal speeds such as "2.5 mm/s [µm]" -> 2.5
speeds = speed_labels.str.extract(r'([\d\.]+)\s*mm/s')[0].astype(float)

# Extract the 5 sample measurements (rows 1–5, columns 1..end)
sample_data = df.iloc[1:6, 1:].astype(float)

# --- 2. Compute statistics ---
avg_thickness = sample_data.mean(axis=0).values
std_thickness = sample_data.std(axis=0, ddof=1).values

# --- 3. Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(speeds, avg_thickness, label="Average thickness")

plt.errorbar(
    speeds,
    avg_thickness,
    yerr=std_thickness,
    fmt="none",
    capsize=5,
    ecolor="black",
    linewidth=1
)

plt.xlabel("Withdrawal speed (mm/s)")
plt.ylabel("Average film thickness (µm)")
plt.title("Average Film Thickness vs Withdrawal Speed")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("thickness_vs_speed.png", dpi=300)
plt.show()