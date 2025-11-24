"""
This module should handle all interactions with the real world
"""

# === Imports ===
from .control_http import DuetHTTP
from .deck_grid import vials, plate, tips, safe_z, surface_z
import jubilee_pipette_bodemo.image_processing as img
import numpy as np
from datetime import date


# === Machine Connection ===
IP = "192.168.1.8"          # your Duet IP
JUBILEE = DuetHTTP(IP)

# Safe travel & hover heights
SAFE_Z = safe_z()           # global travel height (set high enough to clear all holders with tip attached)
HOVER  = 5.0                # mm above surface when approaching


# === Color Mixing Sample Function (from original file) ===
def sample_point(jubilee, pipette, Camera, sample_composition: tuple, sample_volume: float,
                 well, color_stocks, save=True):
    """
    Sample a specified point.

    Inputs:
    jubilee: Jubilee Machine object
    Pipette: Jubilee library pipette object, configured
    Camera: Jubilee library camera tool
    sample_composition (tuple): stock color values - either 0–1 or 0–sample_volume.
    sample_volume: total sample volume
    well: location to prepare sample
    color_stocks: list of color stock positions

    Returns:
        RGB - tuple RGB value of resulting solution
    """
    # Calculate volumes
    if np.round(np.sum(sample_composition)) == 1:
        volumes = [sample_volume * sample for sample in sample_composition]
    elif np.round(np.sum(sample_composition)) == sample_volume:
        volumes = sample_composition
    else:
        print(f'Error: Color composition does not sum to 1 or expected sample volume of {sample_volume}')
        return None

    # Zero out small volumes
    raw_volumes = volumes
    volumes = [vol if vol > 3 else 0 for vol in raw_volumes]
    print('Corrected volumes:', volumes)

    # Identify last color to mix
    stock_to_mix = None
    for i, v in enumerate(volumes):
        if v != 0:
            stock_to_mix = i

    # Pipette colors into well
    jubilee.pickup_tool(pipette)
    pipette.transfer(volumes, color_stocks, well.top(-1),
                     blowout=True, new_tip='once', mix_after=(275, 6, color_stocks[stock_to_mix]))

    # Switch to camera tool
    jubilee.pickup_tool(Camera)
    image = Camera.capture_image(well, light=True, light_intensity=1)

    jubilee.park_tool()

    # Process image
    RGB = img.process_image(image)

    if save:
        td = date.today().strftime("%Y%m%d")
        filename = f"{td}_{well.name}_{well.slot}"
        img.save_image(image, filename)

    return RGB, image


# === Motion Helpers ===
def z_hover(holder_key: str) -> float:
    """Return hover Z at least SAFE_Z or 5 mm above holder surface."""
    return max(SAFE_Z, surface_z(holder_key) + HOVER)


def goto_xy(x: float, y: float, holder_key: str):
    """Go to X,Y above holder safely, using hover height."""
    JUBILEE.move_to(z=z_hover(holder_key), f=2000)
    JUBILEE.move_to(x=x, y=y, f=3000)


def goto_vial(index: int):
    """Hover above vial by 1-based index."""
    x, y = vials(index=index)
    goto_xy(x, y, "scint_vials")


def goto_well(row: str, col: int):
    """Hover above 96-well plate position (e.g., row 'A', col 1)."""
    x, y = plate(row, col)
    goto_xy(x, y, "plate96")


def goto_tip(row: str, col: int):
    """Hover above tip rack position (e.g., row 'A', col 1)."""
    x, y = tips(row, col)
    goto_xy(x, y, "tiprack")
