# run_color_mixing_demo.py (v5.4 - Fixed Dispense & Mixing)
# - Different tip per color + dedicated mixing tip
# - One-shot tip pickup, eject via V-axis to setpoint
# - Single dip/aspirate per vial (no air gap, no prewet)
# - Approach → pause → dive for vials/wells
# - Per-sample imaging + CSV log
# - Per-color volume overrides (global or per-batch)
# - Fixed: Proper V-axis dispense direction and full-volume mixing

import csv
import time
import datetime
import string
from pathlib import Path

import cv2
import numpy as np

from jubilee_pipette_bodemo.jubilee_protocols import JUBILEE, z_hover, surface_z
from jubilee_pipette_bodemo.pipette import Pipette
from jubilee_pipette_bodemo.deck_grid import vials, plate, tips

try:
    import jubilee_pipette_bodemo.image_processing as img
    HAVE_IMG = True
except Exception:
    HAVE_IMG = False

# =====================================================================
# USER PARAMETERS
# =====================================================================

# Machine & Camera
IP = "192.168.1.8"
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Source vials
RED_VIAL, YELLOW_VIAL, BLUE_VIAL = 1, 2, 3

# Tip rack positions (row, col)
RED_TIP_POS    = ("A", 1)
YELLOW_TIP_POS = ("A", 2)
BLUE_TIP_POS   = ("A", 3)
MIX_TIP_POS    = ("A", 4)

# Default mixing ratios (used if not overridden per sample)
# With 272 µL max capacity, these ratios will produce a full well
DEFAULT_RATIOS   = (0.50, 0.30, 0.20)  # R, Y, B (must sum to 1.0)
DEFAULT_TOTAL_UL = 272.0  # Max pipette tip capacity

# Optional: Global per-color volume override (µL)
# Set to None to use ratios, or provide dict like: {"R": 136.0, "Y": 81.6, "B": 54.4}
# WARNING: Total volume must not exceed 272 µL
PER_COLOR_UL = None

# Motion speeds (mm/min)
TRAVEL_F = 3000
Z_F_FAST = 2000
Z_F_SLOW = 800

# Z approach parameters
APPROACH_ABOVE_MM_VIAL = 5.0   # Height above vial before diving
APPROACH_ABOVE_MM_WELL = 5.0   # Height above well before diving
DWELL_MS_BEFORE_DIVE   = 250   # Pause at approach height

# Aspiration/Dispense depths (mm below surface)
ASPIRATE_DEPTH_VIAL_MM = 40.0
DISPENSE_DEPTH_WELL_MM = 10.0

# Aspiration parameters
ASP_OVERDRAW_UL       = 10.0   # Extra volume to aspirate for accuracy
ASP_HOLD_MS           = 200    # Pause after aspiration (ms)
WITHDRAW_AFTER_ASP_MM = 2.0    # Slight withdrawal before exiting vial

# Tool selection
USE_TOOL_FOR_PIPETTE = True
USE_TOOL_FOR_CAMERA  = True
PIPETTE_TOOL_NUM = 0
CAMERA_TOOL_NUM  = 1

# Tip handling
TIP_SEAT_DEPTH_MM = 15.0   # How far to press tip during pickup
V_EJECT_ABS = 65.0         # V position to eject tip
V_HOME_ABS  = 475.0        # V homed/ready position

# Pipette calibration (1.16 µL per 1 mm V-axis travel)
UL_PER_MM = 1.16           # Volume per mm of V-axis travel

# Well indexing
WELL_START      = ("A", 1)
WELL_ORDER_ROWS = list(string.ascii_uppercase[:8])  # A-H
WELL_ORDER_COLS = list(range(1, 13))                # 1-12

# =====================================================================
# BATCH DEFINITION
# =====================================================================

# Maximum tip capacity based on calibration (1.16 µL per 1 mm V-axis travel)
MAX_TIP_CAPACITY_UL = 272.0

def build_batch(n, ratios=DEFAULT_RATIOS, total_ul=DEFAULT_TOTAL_UL):
    """Generate n identical samples with specified ratios/total volume."""
    return [{"ratios": ratios, "total_ul": total_ul} for _ in range(n)]

def build_batch_from_volumes(samples_list):
    """
    Build batch from explicit volume specifications.
    
    Args:
        samples_list: List of tuples (r_ul, y_ul, b_ul) for each sample
        
    Example:
        build_batch_from_volumes([
            (136.0, 81.6, 54.4),  # Sample 1: 50% R, 30% Y, 20% B
            (200.0, 50.0, 22.0),  # Sample 2: different ratio
            (100.0, 100.0, 72.0), # Sample 3: equal R+Y, less B
        ])
    """
    batch = []
    for r_ul, y_ul, b_ul in samples_list:
        total = r_ul + y_ul + b_ul
        if total > MAX_TIP_CAPACITY_UL:
            raise ValueError(f"Total volume {total:.1f} µL exceeds max capacity {MAX_TIP_CAPACITY_UL} µL")
        batch.append({"volumes_ul": (r_ul, y_ul, b_ul)})
    return batch

# Define your batch here - Choose ONE of these methods:

# METHOD 1: Use ratios (all samples identical)
# BATCH = build_batch(3, ratios=(0.50, 0.30, 0.20), total_ul=272.0)

# METHOD 2: Specify exact volumes per sample (ACTIVE)
BATCH = build_batch_from_volumes([
    (136.0, 81.6, 54.4),   # Sample 1: 50% R, 30% Y, 20% B = 272 µL total
    (180.0, 60.0, 32.0),   # Sample 2: 66% R, 22% Y, 12% B = 272 µL total  
    (100.0, 100.0, 72.0),  # Sample 3: 37% R, 37% Y, 26% B = 272 µL total
])

# METHOD 3: Use global per-color override (set PER_COLOR_UL above instead)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _dwell_ms(ms: int):
    """Pause for specified milliseconds."""
    if ms > 0:
        time.sleep(ms / 1000.0)

def _gcode(cmd: str):
    """Send G-code command to Jubilee with debug output."""
    fn = getattr(JUBILEE, "gcode", None)
    if callable(fn):
        print(f"      [G-code] {cmd}")  # Debug: show actual command
        return fn(cmd)
    else:
        print(f"[WARN] JUBILEE.gcode() not available; skipping: {cmd}")

def select_tool(n: int):
    """Select tool number n."""
    _gcode(f"T{n}")

def deselect_tool():
    """Deselect all tools."""
    _gcode("T-1")

# =====================================================================
# CAMERA FUNCTIONS
# =====================================================================

def _grab_frame(index=0, width=1280, height=720):
    """Capture a single frame from camera."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened")
    
    # Discard first few frames for camera stabilization
    for _ in range(3):
        cap.read()
    
    ok, frame = cap.read()
    cap.release()
    
    if not ok or frame is None:
        raise RuntimeError("Failed to grab camera frame")
    return frame

def _masked_mean_rgb(frame_bgr, radius=50):
    """Calculate mean RGB in circular mask at frame center."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius), 255, -1)
    
    # Calculate mean for each channel
    means = []
    for ch in range(3):
        channel_data = frame_bgr[:, :, ch][mask > 0]
        means.append(float(channel_data.mean()) if channel_data.size else 0.0)
    
    return [means[2], means[1], means[0]]  # Convert BGR to RGB

def _jpeg_bytes_from_bgr(frame):
    """Encode frame as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# =====================================================================
# DECK MOVEMENT FUNCTIONS
# =====================================================================

def move_above_vial(vial_index: int):
    """Move to hover position above specified vial."""
    x, y = vials(index=vial_index)
    JUBILEE.move_to(z=z_hover("scint_vials"), f=Z_F_FAST)
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)

def move_above_well(row: str, col: int):
    """Move to hover position above specified well."""
    x, y = plate(row, col)
    JUBILEE.move_to(z=z_hover("plate96"), f=Z_F_FAST)
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)

def move_above_tip(row: str, col: int):
    """Move to hover position above specified tip."""
    x, y = tips(row, col)
    JUBILEE.move_to(z=z_hover("tiprack"), f=Z_F_FAST)
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)

def approach_then_dive(holder_key: str, approach_mm: float, depth_mm: float):
    """Approach surface, pause, then dive to specified depth."""
    z_surface = surface_z(holder_key)
    JUBILEE.move_to(z=z_surface + approach_mm, f=Z_F_SLOW)
    _dwell_ms(DWELL_MS_BEFORE_DIVE)
    JUBILEE.move_to(z=z_surface - depth_mm, f=Z_F_SLOW)

# =====================================================================
# TIP HANDLING
# =====================================================================

def pickup_tip_once(row: str, col: int):
    """Pick up tip with single press."""
    print(f"  Picking up tip at {row}{col}...")
    move_above_tip(row, col)
    JUBILEE.move_to(z=surface_z("tiprack") + 1.0, f=Z_F_SLOW)
    JUBILEE.move_to(z=surface_z("tiprack") - TIP_SEAT_DEPTH_MM, f=Z_F_SLOW)
    JUBILEE.move_to(z=z_hover("tiprack"), f=Z_F_FAST)

def return_tip_to_rack(p: Pipette, row: str, col: int):
    """Return tip to rack and eject using V-axis."""
    print(f"  Ejecting tip at {row}{col}...")
    x, y = tips(row, col)
    
    # Move to socket position
    JUBILEE.move_to(z=z_hover("tiprack"), f=Z_F_FAST)
    JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)
    JUBILEE.move_to(z=surface_z("tiprack") + 1.0, f=Z_F_SLOW)
    
    # Eject tip by pressing V-axis down using G-code (fast eject)
    _gcode(f"G1 V{V_EJECT_ABS} F6000")  # 6000 mm/min for quick eject
    _gcode("M400")  # Wait for moves to complete
    _dwell_ms(100)  # Reduced dwell time
    
    # Retract V-axis to home position (very fast retraction)
    _gcode(f"G1 V{V_HOME_ABS} F8000")  # 8000 mm/min for fast retraction
    _gcode("M400")  # Wait for moves to complete
    
    # Retract Z-axis
    JUBILEE.move_to(z=z_hover("tiprack"), f=Z_F_FAST)

# =====================================================================
# LIQUID HANDLING
# =====================================================================

def aspirate_from_vial(p: Pipette, vial_index: int, microliters: float):
    """
    Single dip aspirate from vial with manual V-axis control.
    V=475 is MAX (home/empty). To aspirate, move DOWN first, then UP.
    Aspiration: V moves from LOW → HIGH (+V direction draws liquid up)
    
    IMPORTANT: V-axis is lowered BEFORE submerging tip to avoid bubbling.
    """
    # Calculate V-axis travel needed
    mm_travel = microliters / UL_PER_MM
    v_start = V_HOME_ABS - mm_travel  # Move DOWN from home to create space
    v_end = V_HOME_ABS                # Aspirate UP to home
    
    print(f"    Aspirating {microliters:.1f} µL: V {v_start:.1f} → {v_end:.1f} (+{mm_travel:.1f} mm)")
    
    # FIRST: Move V DOWN to starting position BEFORE entering liquid
    print(f"    Pre-positioning V-axis before submerging...")
    _gcode(f"G1 V{v_start:.2f} F4000")
    _gcode("M400")
    _dwell_ms(100)
    
    # NOW move to vial and submerge (V is already at low position)
    move_above_vial(vial_index)
    approach_then_dive("scint_vials", APPROACH_ABOVE_MM_VIAL, ASPIRATE_DEPTH_VIAL_MM)
    
    # Now aspirate by moving V UP (draws liquid into tip)
    _gcode(f"G1 V{v_end:.2f} F3000")
    _gcode("M400")
    
    # Hold for liquid settling
    _dwell_ms(ASP_HOLD_MS)
    
    # Withdraw slightly while still in liquid
    if WITHDRAW_AFTER_ASP_MM > 0:
        withdraw_z = surface_z("scint_vials") - (ASPIRATE_DEPTH_VIAL_MM - WITHDRAW_AFTER_ASP_MM)
        JUBILEE.move_to(z=withdraw_z, f=Z_F_SLOW)
    
    # Exit vial
    JUBILEE.move_to(z=surface_z("scint_vials") + APPROACH_ABOVE_MM_VIAL, f=Z_F_SLOW)
    JUBILEE.move_to(z=z_hover("scint_vials"), f=Z_F_FAST)

def dispense_to_well(p: Pipette, row: str, col: int, microliters: float):
    """
    Dispense liquid into well with manual V-axis control.
    V should be at HOME (475) after aspiration with liquid inside.
    Dispensing: V moves from HOME (475) → DOWN (lower value) to push liquid out.
    Includes blowout to ensure all liquid exits the tip.
    
    CRITICAL: Exits well BEFORE returning V to home to avoid re-aspirating liquid.
    """
    move_above_well(row, col)
    approach_then_dive("plate96", APPROACH_ABOVE_MM_WELL, DISPENSE_DEPTH_WELL_MM)
    
    # Calculate V-axis travel needed
    mm_travel = microliters / UL_PER_MM
    
    # After aspiration, V should be at HOME with liquid inside
    v_start = V_HOME_ABS  # 475.0
    v_end = V_HOME_ABS - mm_travel  # Move DOWN (negative direction)
    
    print(f"    Dispensing {microliters:.1f} µL: V {v_start:.1f} → {v_end:.1f} (-{mm_travel:.1f} mm)")
    
    # Dispense by moving V DOWN (negative direction pushes liquid out)
    _gcode(f"G1 V{v_end:.2f} F2000")
    _gcode("M400")
    _dwell_ms(100)
    
    # Blowout - push extra 8mm DOWN to ensure all liquid exits
    blowout_mm = 8.0
    v_blowout = v_end - blowout_mm  # Even further down
    print(f"    Blowout: V {v_end:.1f} → {v_blowout:.1f} (-{blowout_mm:.1f} mm)")
    _gcode(f"G1 V{v_blowout:.2f} F1000")
    _gcode("M400")
    _dwell_ms(150)
    
    # CRITICAL: Exit well BEFORE returning V to home to avoid re-aspirating
    print(f"    Exiting well before homing V-axis...")
    JUBILEE.move_to(z=surface_z("plate96") + APPROACH_ABOVE_MM_WELL, f=Z_F_SLOW)
    JUBILEE.move_to(z=z_hover("plate96"), f=Z_F_FAST)
    
    # NOW return V to home (safe, outside the well)
    print(f"    Homing V-axis (outside well): V {v_blowout:.1f} → {V_HOME_ABS:.1f}")
    _gcode(f"G1 V{V_HOME_ABS:.2f} F4000")
    _gcode("M400")

def mix_in_well(p: Pipette, row: str, col: int, total_ul: float, cycles: int = 3):
    """
    Mix liquid in well by aspirating the ENTIRE volume and dispensing it back.
    Each cycle: move DOWN to create space → move UP to aspirate ALL liquid → 
                move DOWN to dispense ALL liquid → blowout
    
    IMPORTANT: V-axis is pre-positioned BEFORE entering well to avoid bubbling.
    CRITICAL: Exits well BEFORE returning V to home to avoid re-aspirating liquid.
    
    Args:
        total_ul: Total volume in the well (will aspirate ALL of it)
        cycles: Number of mix cycles (default 3)
    """
    # Mix the ENTIRE volume in the well
    mm_travel = total_ul / UL_PER_MM
    
    v_low = V_HOME_ABS - mm_travel   # Bottom position (space for liquid)
    v_high = V_HOME_ABS              # Top position (home)
    
    print(f"    Mixing {cycles} cycles of FULL {total_ul:.1f} µL ({mm_travel:.1f} mm V-travel)")
    
    # Pre-position V-axis to low position BEFORE entering well
    print(f"    Pre-positioning V-axis before entering well...")
    _gcode(f"G1 V{v_low:.2f} F4000")
    _gcode("M400")
    _dwell_ms(100)
    
    # NOW move to well and submerge (V is already at low position)
    move_above_well(row, col)
    approach_then_dive("plate96", APPROACH_ABOVE_MM_WELL, DISPENSE_DEPTH_WELL_MM)
    
    # Perform mixing cycles with full volume
    for i in range(cycles):
        print(f"      Mix cycle {i+1}/{cycles}")
        
        # Aspirate FULL VOLUME by moving UP
        _gcode(f"G1 V{v_high:.2f} F3000")
        _gcode("M400")
        _dwell_ms(100)
        
        # Dispense FULL VOLUME by moving DOWN
        _gcode(f"G1 V{v_low:.2f} F2000")
        _gcode("M400")
        _dwell_ms(50)
    
    # Final blowout - push extra 8mm to clear tip completely
    blowout_mm = 8.0
    v_blowout = v_low - blowout_mm
    print(f"    Final blowout: V {v_low:.1f} → {v_blowout:.1f} (-{blowout_mm:.1f} mm)")
    _gcode(f"G1 V{v_blowout:.2f} F1000")
    _gcode("M400")
    _dwell_ms(150)
    
    # CRITICAL: Exit well BEFORE returning V to home to avoid re-aspirating
    print(f"    Exiting well before homing V-axis...")
    JUBILEE.move_to(z=surface_z("plate96") + APPROACH_ABOVE_MM_WELL, f=Z_F_SLOW)
    JUBILEE.move_to(z=z_hover("plate96"), f=Z_F_FAST)
    
    # NOW return V to home (safe, outside the well)
    print(f"    Homing V-axis: V {v_blowout:.1f} → {V_HOME_ABS:.1f}")
    _gcode(f"G1 V{V_HOME_ABS:.2f} F4000")
    _gcode("M400")

# =====================================================================
# BATCH PROCESSING
# =====================================================================

def iterate_wells(start=("A", 1)):
    """Generate well positions in order starting from specified position."""
    start_r, start_c = start
    started = False
    
    for r in WELL_ORDER_ROWS:
        for c in WELL_ORDER_COLS:
            if not started:
                if r == start_r and c == start_c:
                    started = True
                else:
                    continue
            yield (r, c)

def resolve_volumes_for_item(item):
    """
    Determine R, Y, B volumes for sample in µL.
    Priority: item["volumes_ul"] > PER_COLOR_UL > ratios + total_ul
    
    Validates that total volume does not exceed MAX_TIP_CAPACITY_UL (272 µL).
    """
    # Per-item override
    if "volumes_ul" in item:
        v = tuple(item["volumes_ul"])
        assert len(v) == 3, "volumes_ul must have 3 elements (R, Y, B)"
        r_ul, y_ul, b_ul = float(v[0]), float(v[1]), float(v[2])
    # Global override
    elif PER_COLOR_UL:
        r_ul = float(PER_COLOR_UL["R"])
        y_ul = float(PER_COLOR_UL["Y"])
        b_ul = float(PER_COLOR_UL["B"])
    # Calculate from ratios
    else:
        ratios = tuple(item.get("ratios", DEFAULT_RATIOS))
        total_ul = float(item.get("total_ul", DEFAULT_TOTAL_UL))
        
        assert len(ratios) == 3, "ratios must have 3 elements (R, Y, B)"
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
        
        r_ul = total_ul * ratios[0]
        y_ul = total_ul * ratios[1]
        b_ul = total_ul * ratios[2]
    
    # Validate total volume
    total = r_ul + y_ul + b_ul
    if total > MAX_TIP_CAPACITY_UL:
        raise ValueError(
            f"Total volume {total:.1f} µL exceeds max tip capacity {MAX_TIP_CAPACITY_UL} µL. "
            f"Volumes: R={r_ul:.1f}, Y={y_ul:.1f}, B={b_ul:.1f}"
        )
    
    return r_ul, y_ul, b_ul

def process_color(p: Pipette, color_name: str, tip_pos: tuple, vial_idx: int, 
                  dest: tuple, volume_ul: float):
    """
    Process a single color: pickup tip, aspirate, dispense, eject tip.
    """
    print(f"  {color_name}: {volume_ul:.1f} µL")
    pickup_tip_once(*tip_pos)
    aspirate_from_vial(p, vial_idx, volume_ul)
    dispense_to_well(p, dest[0], dest[1], volume_ul)
    return_tip_to_rack(p, *tip_pos)

# =====================================================================
# MAIN WORKFLOW
# =====================================================================

def main():
    # Initialize pipette
    p = Pipette(IP)
    print("Initializing pipette...")
    p.home()
    time.sleep(0.2)
    
    # Select pipette tool if needed
    if USE_TOOL_FOR_PIPETTE:
        print(f"Selecting pipette tool T{PIPETTE_TOOL_NUM}...\n")
        select_tool(PIPETTE_TOOL_NUM)
    
    # Setup well iterator
    well_iter = iterate_wells(WELL_START)
    
    # Setup CSV logging
    csv_path = Path("color_mixing_log.csv")
    write_header = not csv_path.exists()
    
    with csv_path.open("a", newline="") as f_csv:
        fieldnames = [
            "timestamp", "dest_well", "vol_R", "vol_Y", "vol_B",
            "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
            "camera_index", "frame_size", "tips_used"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        # Process each sample
        for i, item in enumerate(BATCH, 1):
            dest = item.get("dest") or next(well_iter)
            r_ul, y_ul, b_ul = resolve_volumes_for_item(item)
            total_ul = r_ul + y_ul + b_ul
            
            print(f"\n{'='*60}")
            print(f"Sample {i}/{len(BATCH)} → Well {dest[0]}{dest[1]}")
            print(f"Total Volume: {total_ul:.1f} µL ({(total_ul/MAX_TIP_CAPACITY_UL)*100:.1f}% capacity)")
            print(f"{'='*60}")
            
            # RED
            process_color(p, "RED", RED_TIP_POS, RED_VIAL, dest, r_ul)
            
            # YELLOW
            process_color(p, "YELLOW", YELLOW_TIP_POS, YELLOW_VIAL, dest, y_ul)
            
            # BLUE
            process_color(p, "BLUE", BLUE_TIP_POS, BLUE_VIAL, dest, b_ul)
            
            # MIXING - Mix the FULL volume (all 272 µL)
            print(f"  MIXING: Full {total_ul:.1f} µL ({total_ul/UL_PER_MM:.1f} mm = {int(total_ul/UL_PER_MM)} mm)")
            pickup_tip_once(*MIX_TIP_POS)
            mix_in_well(p, dest[0], dest[1], total_ul=total_ul, cycles=3)
            return_tip_to_rack(p, *MIX_TIP_POS)
            
            # IMAGING
            if USE_TOOL_FOR_CAMERA:
                print(f"  Switching to camera tool T{CAMERA_TOOL_NUM}...")
                select_tool(CAMERA_TOOL_NUM)
                x, y = plate(dest[0], dest[1])
                JUBILEE.move_to(z=200, f=Z_F_FAST)
                JUBILEE.move_to(x=x, y=y, f=TRAVEL_F)
                print(f"  Camera positioned at Z=200mm above well {dest[0]}{dest[1]}")
            
            print("  Capturing image...")
            frame = _grab_frame(index=CAM_INDEX, width=FRAME_W, height=FRAME_H)
            
            # Process image for RGB measurement
            if HAVE_IMG and hasattr(img, "process_image"):
                jpeg_bytes = _jpeg_bytes_from_bgr(frame)
                rgb_mean = [float(v) for v in img.process_image(jpeg_bytes)]
            else:
                rgb_mean = _masked_mean_rgb(frame, radius=50)
            
            rgb_rounded = [round(v, 1) for v in rgb_mean]
            
            # Save image
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = Path(f"mix_{dest[0]}{dest[1]}_{ts}.jpg")
            cv2.imwrite(str(img_path), frame)
            print(f"  Saved: {img_path.name} | RGB: {rgb_rounded}")
            
            # Log to CSV
            row = {
                "timestamp": ts,
                "dest_well": f"{dest[0]}{dest[1]}",
                "vol_R": r_ul,
                "vol_Y": y_ul,
                "vol_B": b_ul,
                "rgb_mean_r": rgb_rounded[0],
                "rgb_mean_g": rgb_rounded[1],
                "rgb_mean_b": rgb_rounded[2],
                "camera_index": CAM_INDEX,
                "frame_size": f"{FRAME_W}x{FRAME_H}",
                "tips_used": (f"R:{RED_TIP_POS[0]}{RED_TIP_POS[1]};"
                             f"Y:{YELLOW_TIP_POS[0]}{YELLOW_TIP_POS[1]};"
                             f"B:{BLUE_TIP_POS[0]}{BLUE_TIP_POS[1]};"
                             f"M:{MIX_TIP_POS[0]}{MIX_TIP_POS[1]}")
            }
            writer.writerow(row)
            f_csv.flush()
            
            # Switch back to pipette tool
            if USE_TOOL_FOR_CAMERA and USE_TOOL_FOR_PIPETTE:
                select_tool(PIPETTE_TOOL_NUM)
    
    print(f"\n{'='*60}")
    print("✅ Batch complete! All samples processed.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()