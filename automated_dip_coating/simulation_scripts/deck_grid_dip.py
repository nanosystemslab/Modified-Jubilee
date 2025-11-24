# deck_grid_dip.py
import json, os, pathlib, string
from typing import Tuple

THIS_DIR = pathlib.Path(__file__).resolve().parent
CANDIDATES = []
if os.environ.get("DIP_CONFIG_DIR"):
    CANDIDATES.append(pathlib.Path(os.environ["DIP_CONFIG_DIR"]))
CANDIDATES.append(THIS_DIR)
CANDIDATES.append(pathlib.Path.cwd())

def _deck_path() -> pathlib.Path:
    tried = []
    for base in CANDIDATES:
        p = base / "dip_config.json"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError("Could not find dip_config.json. Tried:\n" + "\n".join(tried))

def _load():
    return json.loads(_deck_path().read_text(encoding="utf-8"))

def safe_z() -> float:
    return float(_load()["safe_z"])

def surface_z(holder_key: str) -> float:
    data = _load()
    if holder_key == "bath":
        return float(data["bath"]["z"])
    return float(data[holder_key]["origin"]["z"])

def z_hover(holder_key: str) -> float:
    data = _load()
    hover_mm = 30.0
    if holder_key == "bath":
        hover_mm = float(data["bath"].get("hover_mm", hover_mm))
        return surface_z("bath") + hover_mm
    else:
        hover_mm = float(data[holder_key].get("hover_mm", hover_mm))
        return surface_z(holder_key) + hover_mm

def _rc_from_rowcol(row, col):
    if isinstance(row, str):
        r = string.ascii_uppercase.index(row.upper()) + 1
    else:
        r = int(row)
    c = int(col)
    return r, c

def _xy_from_grid(holder: dict, row, col) -> Tuple[float, float]:
    r, c = _rc_from_rowcol(row, col)
    org   = holder["origin"]
    pitch = holder["pitch"]
    x_dir = holder.get("x_dir", 1)
    y_dir = holder.get("y_dir", 1)
    cols_axis = holder.get("cols_axis", "x")
    rows_axis = holder.get("rows_axis", "y")
    x = org["x"]
    y = org["y"]
    if cols_axis == "x":
        x += (c - 1) * pitch["x"] * x_dir
    elif cols_axis == "y":
        y += (c - 1) * pitch["y"] * y_dir
    else:
        raise ValueError(f"cols_axis must be 'x' or 'y', got {cols_axis}")
    if rows_axis == "y":
        y += (r - 1) * pitch["y"] * y_dir
    elif rows_axis == "x":
        x += (r - 1) * pitch["x"] * x_dir
    else:
        raise ValueError(f"rows_axis must be 'x' or 'y', got {rows_axis}")
    return float(x), float(y)

def slides_dry(row=None, col=None, index=None):
    d = _load()["slides_dry"]
    if index is not None:
        rows, cols = d["rows"], d["cols"]
        i = int(index) - 1
        r = (i // cols) + 1
        c = (i %  cols) + 1
        return _xy_from_grid(d, r, c)
    return _xy_from_grid(d, row, col)

def slides_wet(row=None, col=None, index=None):
    d = _load()["slides_wet"]
    if index is not None:
        rows, cols = d["rows"], d["cols"]
        i = int(index) - 1
        r = (i // cols) + 1
        c = (i %  cols) + 1
        return _xy_from_grid(d, r, c)
    return _xy_from_grid(d, row, col)

def bath():
    b = _load()["bath"]
    return float(b["x"]), float(b["y"])

def debug_which_json():
    p = _deck_path()
    data = _load()
    print("Loading deck JSON from:", p)
    for key in ("slides_dry", "slides_wet"):
        if key in data:
            print(f"{key}:", data[key])
    if "bath" in data:
        print("bath:", data["bath"])
