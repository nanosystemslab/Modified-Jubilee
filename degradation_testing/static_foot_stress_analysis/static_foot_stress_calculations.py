"""
Comprehensive foot stress analysis for Original vs Modified Jubilee
Addresses thesis committee comment on mass, stress calculations, and yield comparison
"""

import pandas as pd
import os

# --- Output directory ---
output_dir = r"C:\Users\dvier\OneDrive\Documents\Thesis\Vibration Effect on Feet"
output_path = os.path.join(output_dir, "jubilee_foot_stress_analysis.xlsx")

# --- Constants ---
G = 9.81  # m/s^2
LB_TO_KG = 0.45359237
MM2_TO_M2 = 1e-6

# --- Input data ---
systems = {
    "Modified Jubilee": {
        "weight_lb": 41.0,
        "foot_areas_mm2": {"back": 1700.0, "front": 1200.0},
    },
    "Original Jubilee": {
        "weight_lb": 31.2,
        "foot_areas_mm2": {"back": 1404.0, "front": 904.0},
    },
}

# --- Yield stress from degradation testing (UPDATE WITH YOUR VALUES) ---
# Replace these placeholder values with your actual experimental results
YIELD_STRESS_MPA = 25.0  # Example: typical PLA yield stress


def compute_stresses(weight_lb: float, area_back_mm2: float, area_front_mm2: float):
    """
    Calculate static compressive stresses in printer feet
    
    Parameters:
    - weight_lb: System weight in pounds
    - area_back_mm2: Contact area of back foot in mm²
    - area_front_mm2: Contact area of front foot in mm²
    
    Returns dictionary with all calculated values
    """
    # Convert weight to mass
    mass_kg = weight_lb * LB_TO_KG
    
    # Total gravitational force
    total_force_N = mass_kg * G
    
    # Force per foot (assuming equal distribution)
    per_foot_force_N = total_force_N / 4.0
    
    # Convert areas to m²
    area_back_m2 = area_back_mm2 * MM2_TO_M2
    area_front_m2 = area_front_mm2 * MM2_TO_M2
    
    # Compressive stress: σ = F/A
    stress_back_MPa = (per_foot_force_N / area_back_m2) / 1e6
    stress_front_MPa = (per_foot_force_N / area_front_m2) / 1e6
    
    # Worst case: two feet supporting entire weight (front/back tilt)
    worst_case_force_N = 0.5 * total_force_N
    worst_case_back_MPa = (worst_case_force_N / area_back_m2) / 1e6
    worst_case_front_MPa = (worst_case_force_N / area_front_m2) / 1e6
    
    return {
        "mass_kg": mass_kg,
        "total_force_N": total_force_N,
        "per_foot_force_N": per_foot_force_N,
        "stress_back_MPa": stress_back_MPa,
        "stress_front_MPa": stress_front_MPa,
        "worst_case_back_MPa": worst_case_back_MPa,
        "worst_case_front_MPa": worst_case_front_MPa,
    }


def main():
    # === Table 1: System Mass and Forces ===
    mass_force_rows = []
    for name, data in systems.items():
        w_lb = data["weight_lb"]
        m_kg = w_lb * LB_TO_KG
        F_total = m_kg * G
        F_per_foot = F_total / 4.0
        
        mass_force_rows.append({
            "System": name,
            "Weight (lb)": w_lb,
            "Mass (kg)": round(m_kg, 3),
            "Total Force (N)": round(F_total, 2),
            "Force per Foot (N)": round(F_per_foot, 2)
        })
    
    df_mass_force = pd.DataFrame(mass_force_rows)
    
    # === Table 2: Stress Analysis ===
    stress_rows = []
    for name, data in systems.items():
        w_lb = data["weight_lb"]
        A_back = data["foot_areas_mm2"]["back"]
        A_front = data["foot_areas_mm2"]["front"]
        
        res = compute_stresses(w_lb, A_back, A_front)
        
        stress_rows.append({
            "System": name,
            "Back Foot Area (mm²)": A_back,
            "Front Foot Area (mm²)": A_front,
            "σ_back Normal (MPa)": round(res["stress_back_MPa"], 4),
            "σ_front Normal (MPa)": round(res["stress_front_MPa"], 4),
            "σ_back Worst-Case (MPa)": round(res["worst_case_back_MPa"], 4),
            "σ_front Worst-Case (MPa)": round(res["worst_case_front_MPa"], 4),
        })
    
    df_stress = pd.DataFrame(stress_rows)
    
    # === Table 3: Safety Factor Analysis ===
    safety_rows = []
    for name, data in systems.items():
        w_lb = data["weight_lb"]
        A_back = data["foot_areas_mm2"]["back"]
        A_front = data["foot_areas_mm2"]["front"]
        
        res = compute_stresses(w_lb, A_back, A_front)
        
        # Calculate safety factors (yield stress / applied stress)
        sf_back_normal = YIELD_STRESS_MPA / res["stress_back_MPa"]
        sf_front_normal = YIELD_STRESS_MPA / res["stress_front_MPa"]
        sf_back_worst = YIELD_STRESS_MPA / res["worst_case_back_MPa"]
        sf_front_worst = YIELD_STRESS_MPA / res["worst_case_front_MPa"]
        
        safety_rows.append({
            "System": name,
            "Yield Stress (MPa)": YIELD_STRESS_MPA,
            "SF Back (Normal)": round(sf_back_normal, 1),
            "SF Front (Normal)": round(sf_front_normal, 1),
            "SF Back (Worst-Case)": round(sf_back_worst, 1),
            "SF Front (Worst-Case)": round(sf_front_worst, 1),
            "Min Safety Factor": round(min(sf_back_worst, sf_front_worst), 1)
        })
    
    df_safety = pd.DataFrame(safety_rows)
    
    # === Write to Excel with multiple sheets ===
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_mass_force.to_excel(writer, sheet_name='Mass and Forces', index=False)
        df_stress.to_excel(writer, sheet_name='Stress Analysis', index=False)
        df_safety.to_excel(writer, sheet_name='Safety Factors', index=False)
    
    print(f"✓ Analysis complete! Excel file written to:")
    print(f"  {output_path}\n")
    
    # === Display results in console ===
    print("="*70)
    print("TABLE 1: SYSTEM MASS AND FORCES")
    print("="*70)
    print(df_mass_force.to_string(index=False))
    print()
    
    print("="*70)
    print("TABLE 2: COMPRESSIVE STRESS ANALYSIS (σ = F/A)")
    print("="*70)
    print(df_stress.to_string(index=False))
    print()
    
    print("="*70)
    print("TABLE 3: SAFETY FACTOR COMPARISON")
    print(f"(Yield Stress from Degradation Testing: {YIELD_STRESS_MPA} MPa)")
    print("="*70)
    print(df_safety.to_string(index=False))
    print()
    
    print("NOTES:")
    print("- Normal loading: Weight equally distributed across 4 feet")
    print("- Worst-case: Two feet support entire weight (tipping scenario)")
    print("- Safety Factor = Yield Stress / Applied Stress")
    print(f"- UPDATE YIELD_STRESS_MPA (line 27) with your experimental value!")
    print("="*70)


if __name__ == "__main__":
    main()