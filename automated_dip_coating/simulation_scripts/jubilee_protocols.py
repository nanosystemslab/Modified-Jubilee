# jubilee_protocols.py
# Minimal Duet/RRF 3.x client used by dip-coating scripts.
# - Sends G-code via HTTP /machine/code (RRF 3.x) with fallback to legacy /rr_gcode
# - Provides JUBILEE.gcode(), JUBILEE.move_to(), JUBILEE.home()

from __future__ import annotations
import os, time
from typing import Optional

try:
    import requests  # pip install requests
except ImportError as e:
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`") from e


DEFAULT_HOST = os.getenv("JUBILEE_DUET_HOST", "192.168.1.8")  # set your Duet IP here or via env var

class _DuetHTTP:
    def __init__(self, host: str):
        # Normalize host like "http://<ip>"
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        self.host = host.rstrip("/")
        # RRF 3.x endpoint
        self.code_url = f"{self.host}/machine/code"
        # Legacy (rr_*) endpoints (RRF 2 / some 3.x still support)
        self.rr_gcode_url = f"{self.host}/rr_gcode"

    def _post_code_rrf3(self, code_line: str) -> bool:
        # /machine/code takes JSON {"code": "G28"} in 3.x
        try:
            r = requests.post(self.code_url, json={"code": code_line}, timeout=3)
            return r.ok
        except Exception:
            return False

    def _get_code_legacy(self, code_line: str) -> bool:
        # Legacy: /rr_gcode?gcode=G28
        try:
            r = requests.get(self.rr_gcode_url, params={"gcode": code_line}, timeout=3)
            return r.ok
        except Exception:
            return False

    def send_line(self, code_line: str) -> None:
        code_line = code_line.strip()
        if not code_line:
            return
        # try RRF 3.x
        if self._post_code_rrf3(code_line):
            return
        # fallback
        if self._get_code_legacy(code_line):
            return
        raise RuntimeError(f"Failed to send code to Duet: {code_line!r} (host={self.host})")


class _Jubilee:
    """
    Minimal motion + G-code helper.
    """
    def __init__(self, host: str = DEFAULT_HOST):
        self.duet = _DuetHTTP(host)
        # ensure absolute positioning & mm units (idempotent)
        self.gcode("G90\nG21")

    def gcode(self, code: str) -> None:
        """
        Send raw G-code. Supports multi-line strings.
        """
        for line in code.splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            self.duet.send_line(line)

    def home(self) -> None:
        self.gcode("G28")

    def move_to(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        f: Optional[float] = None,
    ) -> None:
        """
        Absolute move. Any of x/y/z/f can be omitted.
        Units: mm and mm/min (G21 + G90 are set in __init__).
        """
        parts = ["G1"]
        if x is not None: parts.append(f"X{x:.3f}")
        if y is not None: parts.append(f"Y{y:.3f}")
        if z is not None: parts.append(f"Z{z:.3f}")
        if f is not None: parts.append(f"F{f:.0f}")
        self.gcode(" ".join(parts))
        self.gcode("M400")  # Wait for move to complete
        time.sleep(0.1)  # Additional small delay to ensure command processing


# The scripts import JUBILEE (class *instance* or class?)
# They use it like `JUBILEE.move_to(...)`, so export a singleton instance.
JUBILEE = _Jubilee()