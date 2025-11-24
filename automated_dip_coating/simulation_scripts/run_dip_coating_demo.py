#!/usr/bin/env python3
"""
Vacuum-based dip-coating batch flow (side-pickup slides):
 1) select vacuum tool (T1) if not already active
 2) move to slide on DRY tray with 1mm Y-offset
 3) advance 1mm in Y to contact slide
 4) suction, then Z-only lift to safe_z()
 5) XY to bath, go to z_hover("bath")
 6) dip with per-sample submersion time + withdrawal speed
 7) safe_z() -> XY to WET tray -> lower Z -> release
 8) return to safe_z(), ensure valve off
 9) repeat for N_SAMPLES (column-major order)
Assumes axes are already homed in Duet before running.
"""

import time, datetime
from pathlib import Path
import sys, pathlib

# Local-only import of JUBILEE + deck helpers
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from jubilee_protocols import JUBILEE  # local file only
from deck_grid_dip import (
    safe_z, surface_z, z_hover,
    slides_dry, slides_wet, bath
)

# ----------------- USER PARAMETERS -----------------
TRAVEL_F     = 3000          # XY travel speed (mm/min)
Z_F_FAST     = 2000          # Z fast (mm/min)
Z_F_SLOW     = 800           # Z slow (mm/min)

# Pauses (seconds)
PAUSE_BEFORE_Y_ADVANCE  = 2.0    # pause at Z approach height before Y-offset movement
PAUSE_AT_OFFSET_POSITION = 2.0   # pause at Y-offset position before advancing to contact
PAUSE_BEFORE_PICK   = 2.0        # pause after contacting slide before suction
PAUSE_AFTER_SUCTION = 2.0        # pause after turning on suction
PAUSE_BEFORE_DIP    = 2.0        # pause above bath before diving
PAUSE_AFTER_DIP     = 2.0        # pause after withdrawing from bath
PAUSE_BEFORE_RELEASE= 5.0        # pause at position before releasing vacuum
PAUSE_AFTER_RELEASE = 2.0        # pause after releasing before retracting
PAUSE_AT_SAFE_Z     = 5.0        # pause at safe Z before final valve off

# Global defaults if per-sample lists omitted/short
DIP_DEPTH_MM       = 10.0     # below bath surface
SUBMERSION_TIME_S  = 5.0
WITHDRAW_F_DEFAULT = 300      # mm/min

# Per-sample overrides (optional). If shorter than N_SAMPLES, remaining use defaults.
SUBMERSION_TIMES = [5,5,6,6,7,7,8,8,9,9]
WITHDRAW_SPEEDS  = [300,350,300,350,400,450,500,550,600,650]

# GRID SIZE (total rows/cols in your trays)
TRAY_ROWS = 10                # total rows
TRAY_COLS = 2                 # total columns

# Start slot (NOTE: 1-based indexing)
START_ROW = 1
START_COL = 1

N_SAMPLES = 5                # you plan 10 slides per batch

# Tool config
USE_TOOL = True
TOOL_NUM = 1                  # T1 = vacuum tool

# Approach clearances
APP_TRAY_MM = 5.0             # Z approach above tray surface before final Z
APP_BATH_MM = 5.0             # Z approach above bath surface before diving

# Side-pickup Y-offset (1mm back from slide contact)
SLIDE_Y_OFFSET = 10.0          # mm to stay back before contacting slide
PICK_Z_OFFSET  = 0.0
PLACE_Z_OFFSET = 0.0
# ---------------------------------------------------

# Valve controls (fan mapped to out5 as F1)
# NOTE: Valve logic is INVERTED for vacuum pickup
# - CLOSED (S0) = vacuum holds slide during pickup/transport
# - OPEN (S1.0) = vacuum releases slide during dropoff
def vacuum_close():  
    """Close valve = Hold slide (vacuum ON)"""
    print("    [VALVE CLOSE - VACUUM HOLD]")
    JUBILEE.gcode("M106 P1 S0")
    time.sleep(2.0)  # Allow valve to fully close and vacuum to build

def vacuum_open(): 
    """Open valve = Release slide (vacuum OFF)"""
    print("    [VALVE OPEN - VACUUM RELEASE]")
    JUBILEE.gcode("M106 P1 S1.0")
    time.sleep(2.0)  # Allow valve to fully open and vacuum to release

def _sleep_s(s: float):
    if s > 0: 
        print(f"    [PAUSE START: {s}s]")
        time.sleep(s)
        print(f"    [PAUSE END: {s}s]")

def ensure_tool_selected(n: int):
    JUBILEE.gcode(f"T{n}")

# --------- Tray/Bath helpers (side-pickup semantics) ----------
def move_to_safe_z():
    JUBILEE.move_to(z=safe_z(), f=Z_F_FAST)
    time.sleep(2.0)  # Allow buffer to drain and movement to complete

def move_above_dry(rc):
    """Move to position 1mm back from slide (Y-offset)"""
    r, c = rc
    x, y = slides_dry(r, c)
    print(f"  -> DRY[{r},{c}] XY target = ({x:.3f},{y:.3f}) with {SLIDE_Y_OFFSET}mm Y-offset")
    move_to_safe_z()
    JUBILEE.move_to(x=x, y=y - SLIDE_Y_OFFSET, f=TRAVEL_F)
    time.sleep(1.5)  # Allow buffer to drain and movement to complete

def move_above_wet(rc):
    r, c = rc
    x, y = slides_wet(r, c)
    print(f"  -> WET[{r},{c}] XY target = ({x:.3f},{y:.3f})")
    move_to_safe_z()
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)
    time.sleep(1.5)  # Allow buffer to drain and movement to complete

def move_above_bath():
    x, y = bath()
    print(f"  -> BATH XY target = ({x:.3f},{y:.3f})")
    move_to_safe_z()
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)
    time.sleep(1.5)  # Allow buffer to drain and movement to complete
    JUBILEE.move_to(z=z_hover("bath"), f=Z_F_FAST)
    time.sleep(1.0)  # Allow buffer to drain and movement to complete

def pickup_slide_from_dry(rc):
    r, c = rc
    print(f"  Pickup dry[{r},{c}]")
    
    # Close valve FIRST to stop air flow before approaching slide
    print(f"  Closing valve to stop air flow...")
    vacuum_close()
    
    # Move to Y-offset position (1mm back from slide) at safe Z
    move_above_dry(rc)
    print(f"  At Y-offset position, pausing...")
    _sleep_s(PAUSE_AT_OFFSET_POSITION)
    
    # Lower Z directly to final pickup position BEFORE Y advance
    print(f"  Lowering to pickup height...")
    JUBILEE.move_to(z=surface_z('slides_dry') - PICK_Z_OFFSET, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain
    print(f"  At pickup height, pausing before Y advance...")
    _sleep_s(PAUSE_BEFORE_Y_ADVANCE)
    
    # Now advance 1mm in Y to contact slide (already at correct Z)
    x, y = slides_dry(r, c)
    print(f"  Advancing {SLIDE_Y_OFFSET}mm FORWARD in Y to contact slide...")
    JUBILEE.move_to(y=y, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain
    print(f"  Contacted slide, pausing...")
    _sleep_s(PAUSE_BEFORE_PICK)
    
    # Valve is already closed (vacuum holding)
    print(f"  Vacuum engaged (valve already closed), pausing...")
    _sleep_s(PAUSE_AFTER_SUCTION)
    
    # Z-only extraction
    print(f"  Extracting slide...")
    move_to_safe_z()

def place_slide_to_wet(rc):
    r, c = rc
    print(f"  Place wet[{r},{c}]")
    move_above_wet(rc)
    
    # Lower to approach height then to placement position
    print(f"  Lowering to placement position...")
    JUBILEE.move_to(z=surface_z('slides_wet') + APP_TRAY_MM, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain
    JUBILEE.move_to(z=surface_z('slides_wet') - PLACE_Z_OFFSET, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain
    
    # Pause at correct position BEFORE releasing vacuum
    print(f"  At placement position, pausing before valve open...")
    _sleep_s(PAUSE_BEFORE_RELEASE)
    
    # Open valve to release slide (vacuum deactivated)
    # vacuum_open() already has 2.0s delay built in
    vacuum_open()
    _sleep_s(PAUSE_AFTER_RELEASE)
    
    # Retract to safe Z
    print(f"  Retracting to safe Z...")
    move_to_safe_z()
    print(f"  At safe Z, pausing before final valve open...")
    _sleep_s(PAUSE_AT_SAFE_Z)
    
    # Ensure valve is open (vacuum off)
    # vacuum_open() already has 2.0s delay built in
    vacuum_open()

def approach_then_dive_bath(depth_mm: float):
    z_surf = surface_z("bath")
    JUBILEE.move_to(z=z_surf + APP_BATH_MM, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain
    JUBILEE.move_to(z=z_surf - depth_mm, f=Z_F_SLOW)
    time.sleep(1.0)  # Allow buffer to drain

def dip_in_bath(sample_idx: int):
    # Per-sample params with safe fallbacks
    submersion_s = SUBMERSION_TIMES[sample_idx] if sample_idx < len(SUBMERSION_TIMES) else SUBMERSION_TIME_S
    withdraw_f   = WITHDRAW_SPEEDS[sample_idx]  if sample_idx < len(WITHDRAW_SPEEDS)  else WITHDRAW_F_DEFAULT
    print(f"  Dip: depth={DIP_DEPTH_MM} mm, submerge {submersion_s}s, withdraw @ {withdraw_f} mm/min")

    move_above_bath()
    _sleep_s(PAUSE_BEFORE_DIP)
    approach_then_dive_bath(DIP_DEPTH_MM)
    _sleep_s(submersion_s)
    z_exit = surface_z("bath") + APP_BATH_MM
    JUBILEE.move_to(z=z_exit, f=withdraw_f)
    time.sleep(1.5)  # Allow buffer to drain (withdrawal is slower)
    _sleep_s(PAUSE_AFTER_DIP)
    move_to_safe_z()

def iter_tray_positions(n: int, start_r=1, start_c=1):
    """
    Yield 1-based (row,col) pairs in COLUMN-MAJOR order until n positions produced.
    Goes down each column before moving to the next column.
    Covers the full TRAY_ROWS x TRAY_COLS grid (inclusive).
    """
    count = 0
    for c in range(start_c, TRAY_COLS + 1):
        for r in range(start_r if c == start_c else 1, TRAY_ROWS + 1):
            if count >= n:
                return
            yield r, c
            count += 1

def main():
    if USE_TOOL:
        ensure_tool_selected(TOOL_NUM)
    
    # Initialize valve to OPEN (released state)
    vacuum_open()

    log_path = Path("dip_coating_log.csv")
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "ts","idx","dry_r","dry_c","wet_r","wet_c",
            "dip_depth_mm","submersion_s","withdraw_mm_min"
        ])
        if write_header: writer.writeheader()

        for i, rc in enumerate(iter_tray_positions(N_SAMPLES, START_ROW, START_COL)):
            dry_r, dry_c = rc
            wet_r, wet_c = rc  # 1:1 mapping by default
            print(f"\n=== Sample {i+1}/{N_SAMPLES}: dry[{dry_r},{dry_c}] -> bath -> wet[{wet_r},{wet_c}] ===")

            pickup_slide_from_dry(rc)
            dip_in_bath(i)
            place_slide_to_wet((wet_r, wet_c))

            submersion_s = SUBMERSION_TIMES[i] if i < len(SUBMERSION_TIMES) else SUBMERSION_TIME_S
            withdraw_f   = WITHDRAW_SPEEDS[i]  if i < len(WITHDRAW_SPEEDS)  else WITHDRAW_F_DEFAULT
            writer.writerow({
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "idx": i+1,
                "dry_r": dry_r, "dry_c": dry_c,
                "wet_r": wet_r, "wet_c": wet_c,
                "dip_depth_mm": DIP_DEPTH_MM,
                "submersion_s": submersion_s,
                "withdraw_mm_min": withdraw_f
            })
            f.flush()

    print("\n✅ Dip-coating batch complete.")

if __name__ == "__main__":
    main()