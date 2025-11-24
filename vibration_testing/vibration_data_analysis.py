#!/usr/bin/env python3
"""
Raw FFT of Acceleration Data - Direct FFT of time series
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sensor configuration
PLACEMENT_KEYS = {"Direct": "Bed Plate", "Ch0": "Vertical Extrusion", "Ch1": "Tool Head"}
SENSOR_COLORS = {"Direct": "blue", "Ch0": "red", "Ch1": "green"}

def find_time_col(df):
    """Find the time column"""
    for c in df.columns:
        if "time" in c.lower() or "timestamp" in c.lower():
            return c
    raise ValueError("No time column found")

def to_seconds(series):
    """Convert time column to seconds safely."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().all():
        s = s - s.iloc[0]
        if s.iloc[-1] > 0 and np.all(np.diff(s) > 0):
            return s
        if s.iloc[-1] > 1000:
            return s / 1000.0
    try:
        dt = pd.to_datetime(series, errors="raise")
        return (dt - dt.iloc[0]).dt.total_seconds()
    except Exception:
        raise ValueError("Time column format not recognized")

def accel_magnitude(df, key):
    """Calculate acceleration magnitude"""
    cols = [c for c in df.columns if key in c]
    mag = [c for c in cols if "mag" in c.lower()]
    if mag:
        return df[mag[0]].to_numpy()
    xyz = [c for c in cols if c.lower().endswith(("_x","_y","_z"))]
    if len(xyz) == 3:
        return np.linalg.norm(df[xyz].to_numpy(), axis=1)
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy()
    return np.zeros(len(df))

def create_raw_fft_plot(accel_data, time_series, fs, output_dir):
    """Create raw FFT plot of acceleration data"""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    # Convert m/s² to g for analysis (1 g = 9.80665 m/s²)
    g_to_ms2 = 9.80665
    
    plt.figure(figsize=(14, 8))
    
    # Skip first 5 seconds to avoid startup transients
    skip_samples = int(5.0 * fs)
    if skip_samples >= len(time_series):
        skip_samples = 0
        print("Warning: Data too short to skip 5 seconds, using all data")
    
    # Store peak frequencies for text box
    peak_info = {}
    
    for key, label in PLACEMENT_KEYS.items():
        if key in accel_data and len(accel_data[key]) > 0:
            # Get data after startup period
            accel_raw = accel_data[key][skip_samples:]
            
            if len(accel_raw) == 0:
                continue
            
            # Convert to g
            accel_g = accel_raw / g_to_ms2
            
            # Calculate FFT
            N = len(accel_g)
            fft_vals = np.fft.fft(accel_g)
            fft_magnitude = np.abs(fft_vals)
            
            # Create frequency array (only positive frequencies)
            frequencies = np.fft.fftfreq(N, d=1/fs)
            
            # Take only positive frequencies (first half)
            positive_freq_idx = frequencies >= 0
            freq_positive = frequencies[positive_freq_idx]
            fft_positive = fft_magnitude[positive_freq_idx]
            
            # Plot FFT magnitude
            plt.plot(freq_positive, fft_positive, color=SENSOR_COLORS[key], 
                    linewidth=1.5, label=label, alpha=0.8)
            
            # Find peak frequency (excluding DC component at 0 Hz)
            if len(freq_positive) > 1:
                non_dc_idx = freq_positive > 0.5  # Exclude very low frequencies
                if np.any(non_dc_idx):
                    peak_idx = np.argmax(fft_positive[non_dc_idx])
                    peak_freq = freq_positive[non_dc_idx][peak_idx]
                    peak_magnitude = fft_positive[non_dc_idx][peak_idx]
                    
                    # Mark the peak
                    plt.plot(peak_freq, peak_magnitude, 'o', color=SENSOR_COLORS[key], 
                            markersize=8, markeredgecolor='black', markeredgewidth=2)
                    
                    # Add frequency annotation
                    plt.annotate(f'{peak_freq:.1f} Hz', (peak_freq, peak_magnitude), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontsize=9, color=SENSOR_COLORS[key], fontweight='bold')
                    
                    peak_info[key] = (peak_freq, peak_magnitude)
    
    # Create text box with peak information
    if peak_info:
        peak_text = "Peak Frequencies:\n"
        for key, label in PLACEMENT_KEYS.items():
            if key in peak_info:
                freq, magnitude = peak_info[key]
                peak_text += f"{label}: {freq:.1f} Hz\n"
        
        plt.text(0.02, 0.98, peak_text.strip(), transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                 verticalalignment='top', fontsize=10)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Magnitude (g)')
    plt.title('Raw FFT of Acceleration Data (after 5s startup)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, fs/2)  # Show up to Nyquist frequency
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/raw_fft.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_fft_analysis(accel_data, time_series, fs):
    """Print FFT analysis results"""
    print("\n" + "="*60)
    print("RAW FFT ANALYSIS RESULTS")
    print("="*60)
    print(f"Sampling frequency: {fs:.1f} Hz")
    print(f"Duration: {time_series[-1]:.1f} seconds")
    print(f"FFT analysis performed after 5.0s startup period")
    print(f"Input data units: m/s² (converted to g for display)")
    
    # Conversion factor
    g_to_ms2 = 9.80665
    
    # Skip first 5 seconds
    skip_samples = int(5.0 * fs)
    if skip_samples >= len(time_series):
        skip_samples = 0
    
    print(f"Samples used for FFT: {len(time_series) - skip_samples}")
    print(f"Frequency resolution: {fs/(len(time_series) - skip_samples):.3f} Hz")
    
    print(f"\nPEAK FREQUENCY ANALYSIS:")
    
    for key, label in PLACEMENT_KEYS.items():
        if key in accel_data and len(accel_data[key]) > 0:
            # Get data after startup
            accel_raw = accel_data[key][skip_samples:]
            
            if len(accel_raw) == 0:
                continue
                
            # Convert to g
            accel_g = accel_raw / g_to_ms2
            
            # Calculate FFT
            N = len(accel_g)
            fft_vals = np.fft.fft(accel_g)
            fft_magnitude = np.abs(fft_vals)
            frequencies = np.fft.fftfreq(N, d=1/fs)
            
            # Positive frequencies only
            positive_freq_idx = frequencies >= 0
            freq_positive = frequencies[positive_freq_idx]
            fft_positive = fft_magnitude[positive_freq_idx]
            
            print(f"\n  {label}:")
            
            # Find top 3 peaks (excluding DC)
            non_dc_idx = freq_positive > 0.5
            if np.any(non_dc_idx):
                freq_non_dc = freq_positive[non_dc_idx]
                fft_non_dc = fft_positive[non_dc_idx]
                
                # Find peaks
                peak_indices = []
                sorted_indices = np.argsort(fft_non_dc)[::-1]
                
                for i in sorted_indices[:5]:  # Check top 5
                    freq_val = freq_non_dc[i]
                    # Make sure peaks are separated by at least 2 Hz
                    if not any(abs(freq_val - freq_non_dc[j]) < 2.0 for j in peak_indices):
                        peak_indices.append(i)
                    if len(peak_indices) >= 3:
                        break
                
                for i, peak_idx in enumerate(peak_indices, 1):
                    freq_val = freq_non_dc[peak_idx]
                    magnitude_val = fft_non_dc[peak_idx]
                    print(f"    Peak {i}: {freq_val:.2f} Hz (Magnitude: {magnitude_val:.4f} g)")
            
            # Overall statistics
            rms_g = np.sqrt(np.mean(accel_g**2))
            max_g = np.max(np.abs(accel_g))
            print(f"    RMS: {rms_g:.4f} g ({rms_g * g_to_ms2:.3f} m/s²)")
            print(f"    Max: {max_g:.4f} g ({max_g * g_to_ms2:.3f} m/s²)")

def process_file(path, args):
    """Process a single CSV file"""
    print(f"Processing: {path}")
    df = pd.read_csv(path)
    tcol = find_time_col(df)
    time_series = to_seconds(df[tcol]).to_numpy()
    
    # Calculate sampling frequency
    dt = np.median(np.diff(time_series))
    fs = 1.0 / dt
    
    # Extract acceleration data for each sensor
    accel_data = {}
    for key in PLACEMENT_KEYS.keys():
        accel_data[key] = accel_magnitude(df, key)
    
    print(f"Sampling rate: {fs:.1f} Hz, Duration: {time_series[-1]:.1f}s")
    
    # Create output directory
    outdir = f"raw_fft_{os.path.splitext(os.path.basename(path))[0]}"
    
    # Create raw FFT plot and print analysis
    create_raw_fft_plot(accel_data, time_series, fs, outdir)
    print_fft_analysis(accel_data, time_series, fs)

def main():
    parser = argparse.ArgumentParser(description="Raw FFT Analysis of Acceleration Data")
    parser.add_argument("--input", required=True, help="CSV file or folder")
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_file(args.input, args)
    elif os.path.isdir(args.input):
        csv_files = [f for f in os.listdir(args.input) if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {args.input}")
            return
        for fname in csv_files:
            process_file(os.path.join(args.input, fname), args)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()