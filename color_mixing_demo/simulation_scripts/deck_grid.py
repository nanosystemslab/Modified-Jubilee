# deck_grid.py
import json, os, pathlib, string

# --- Locate your deck_grid.json ---
# Repo root = two levels up from this file: src/jubilee_pipette_bodemo -> jubilee_pipette_BOdemo
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_DIR_REPO = REPO_ROOT / "science_jubilee_config"          # where your JSON currently lives
CFG_DIR_ENV  = os.environ.get("SCIENCE_JUBILEE_CONFIG_DIR")   # optional override via env var
CFG_DIR_LOCAL = pathlib.Path(__file__).with_name("science_jubilee_config")  # fallback next to this file

# Try in this order: env var, repo root, local next to module
CANDIDATES = [pathlib.Path(p) for p in [CFG_DIR_ENV, CFG_DIR_REPO, CFG_DIR_LOCAL] if p]

def _deck_path() -> pathlib.Path:
    for base in CANDIDATES:
        p = base / "deck_grid.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find deck_grid.json. Tried:\n" +
        "\n".join(str(base / "deck_grid.json") for base in CANDIDATES)
    )

def _load():
    return json.loads(_deck_path().read_text(encoding="utf-8"))

# --- Public helpers ---

def safe_z() -> float:
    return _load()["safe_z"]

def surface_z(holder_key: str) -> float:
    """holder_key: 'scint_vials', 'plate96', or 'tiprack'"""
    return _load()[holder_key]["origin"]["z"]

def _rc_from_rowcol(row, col):
    """row can be 'A'.. or 1.. ; col is 1-based int"""
    if isinstance(row, str):
        r = string.ascii_uppercase.index(row.upper()) + 1
    else:
        r = int(row)
    c = int(col)
    return r, c

def _xy_from_grid(holder: dict, row, col):
    """Compute X,Y for a grid holder at (row,col)."""
    r, c = _rc_from_rowcol(row, col)
    org   = holder["origin"]
    pitch = holder["pitch"]

    # directions (+1 or -1)
    x_dir = holder.get("x_dir", 1)
    y_dir = holder.get("y_dir", 1)

    # which axis each index advances along
    cols_axis = holder.get("cols_axis", "x")  # "x" or "y"
    rows_axis = holder.get("rows_axis", "y")  # "y" or "x"

    # start at A1 origin
    x = org["x"]
    y = org["y"]

    # apply column offset
    if cols_axis == "x":
        x += (c - 1) * pitch["x"] * x_dir
    elif cols_axis == "y":
        y += (c - 1) * pitch["y"] * y_dir
    else:
        raise ValueError(f"cols_axis must be 'x' or 'y', got {cols_axis}")

    # apply row offset
    if rows_axis == "y":
        y += (r - 1) * pitch["y"] * y_dir
    elif rows_axis == "x":
        x += (r - 1) * pitch["x"] * x_dir
    else:
        raise ValueError(f"rows_axis must be 'x' or 'y', got {rows_axis}")

    return x, y

def vials(row=None, col=None, index=None):
    d = _load()["scint_vials"]
    if index is not None:
        rows, cols = d["rows"], d["cols"]
        i = int(index) - 1
        r = (i // cols) + 1
        c = (i %  cols) + 1
        return _xy_from_grid(d, r, c)
    return _xy_from_grid(d, row, col)

def plate(row, col):
    return _xy_from_grid(_load()["plate96"], row, col)

def tips(row, col):
    return _xy_from_grid(_load()["tiprack"], row, col)

# Optional: quick debug helper
def debug_which_json():
    p = _deck_path()
    data = _load()
    print("Loading deck JSON from:", p)
    for key in ("scint_vials", "plate96", "tiprack"):
        if key in data:
            h = data[key]
            print(
                f"{key}: origin={h['origin']} pitch={h['pitch']} "
                f"dirs=({h.get('x_dir',1)},{h.get('y_dir',1)}) "
                f"axes=({h.get('cols_axis','x')},{h.get('rows_axis','y')})"
            )
