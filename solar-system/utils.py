"""Workshop contingency helpers ONLY — no scientific wrappers.

All TAP (``pyvo``), occultation (``lineaSSP`` / ``requests``), and
matplotlib logic must live explicitly in the notebook / tutorial Markdown.
This module only probes connectivity and loads packaged files under ``data/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

OCCULTATION_API_BASE = "https://solarsystem.linea.org.br/api"
TAP_BASE_URL = "https://userquery.linea.org.br/tap"

OFFLINE_MODE: bool = False


def set_offline_mode(value: bool) -> None:
    """Force offline/fallback behaviour for subsequent contingency loads."""
    global OFFLINE_MODE
    OFFLINE_MODE = bool(value)


def probe_services(timeout_s: float = 8.0) -> dict[str, bool]:
    """Ping Occultation API and TAP. Sets ``OFFLINE_MODE`` if either fails."""
    global OFFLINE_MODE
    status: dict[str, bool] = {"occultation_api": False, "tap": False}

    try:
        r = requests.get(
            f"{OCCULTATION_API_BASE}/asteroids/count/",
            timeout=timeout_s,
            headers={"Accept": "application/json"},
        )
        status["occultation_api"] = bool(r.ok)
    except requests.RequestException:
        status["occultation_api"] = False

    try:
        # Lightweight capability check (same host used by pyvo TAPService).
        r = requests.get(f"{TAP_BASE_URL}/capabilities", timeout=timeout_s)
        status["tap"] = bool(r.ok) or r.status_code in {200, 400, 405}
        if not status["tap"]:
            # Fallback probe: sync endpoint presence
            r2 = requests.post(
                f"{TAP_BASE_URL}/sync",
                data={"LANG": "PostgreSQL", "REQUEST": "doQuery", "QUERY": "SELECT 1 AS ok"},
                timeout=timeout_s,
            )
            status["tap"] = bool(r2.ok) or ("VOTABLE" in r2.text.upper()) or ("<TABLE" in r2.text)
    except requests.RequestException:
        status["tap"] = False

    OFFLINE_MODE = not (status["occultation_api"] and status["tap"])
    status["offline_mode"] = OFFLINE_MODE
    return status


def _load_table(path: Path) -> pd.DataFrame:
    """Load CSV/Parquet; if Parquet needs missing ``pyarrow``, try sibling ``.csv``."""
    if not path.exists():
        # Prefer CSV twin when only parquet was requested but is absent.
        csv_twin = path.with_suffix(".csv")
        if csv_twin.exists():
            return pd.read_csv(csv_twin)
        raise FileNotFoundError(f"Fallback file missing: {path}")
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            csv_twin = path.with_suffix(".csv")
            if csv_twin.exists():
                print(
                    f"[fallback] parquet engine missing ({exc}); "
                    f"loading {csv_twin.name} instead. "
                    "Install with: pip install pyarrow"
                )
                return pd.read_csv(csv_twin)
            raise ImportError(
                "Reading .parquet requires pyarrow (or fastparquet). "
                "Install: pip install pyarrow   — or use the sibling .csv fallback."
            ) from exc
    return pd.read_csv(path)


def load_fallback_occultations(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load packaged occultation predictions (CSV) when the live API is down."""
    local = Path(path) if path is not None else (DATA_DIR / "occultations_sample.csv")
    set_offline_mode(True)
    return _load_table(local)


def load_fallback_photometry(
    target: str = "vesta",
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load packaged photometry when TAP is down.

    Parameters
    ----------
    target
        ``\"vesta\"`` → ``data/vesta_photometry.parquet``;
        ``\"eros\"`` → ``data/eros_photometry.parquet``.
    path
        Optional explicit file path overriding ``target``.
    """
    if path is not None:
        local = Path(path)
    else:
        key = str(target).strip().lower()
        mapping = {
            "vesta": DATA_DIR / "vesta_photometry.parquet",
            "4": DATA_DIR / "vesta_photometry.parquet",
            "eros": DATA_DIR / "eros_photometry.parquet",
            "433": DATA_DIR / "eros_photometry.parquet",
        }
        if key not in mapping:
            raise ValueError("target must be 'vesta'/'4' or 'eros'/'433' (or pass path=...).")
        local = mapping[key]
    set_offline_mode(True)
    return _load_table(local)


def load_fallback_csv(name: str) -> pd.DataFrame:
    """Load any packaged CSV under ``data/`` by file name (contingency only)."""
    local = DATA_DIR / name
    set_offline_mode(True)
    return _load_table(local)
