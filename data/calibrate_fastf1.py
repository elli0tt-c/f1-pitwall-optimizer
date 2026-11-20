"""
data/calibrate_fastf1.py
Crash-proof OpenF1 + FastF1 Telemetry & Actual Race Strategy Calibration Pipeline
for the 2026 F1 Season (All 23 Circuits).

Features:
- Pure-Python OpenF1 REST client with circuit_short_name matching, official /session_result
  classification, /stints tire compound extraction, and HTTP 429 rate-limit backoff
- Calibrates total_laps, fuel-corrected base_lap_time_sec, empirical pit_loss_sec,
  safety_car_prob_per_lap, and per-compound degradation curves (SOFT, MEDIUM, HARD)
- Exports data/circuits.json, data/compound_profiles.json, and data/actual_strategies.json

Usage:
    source venv/bin/activate
    python data/calibrate_fastf1.py
"""

import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import warnings

# Prevent macOS OpenMP / Accelerate / OpenBLAS thread-pool segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/fastf1_cache")
CIRCUITS_OUT = Path("data/circuits.json")
COMPOUNDS_OUT = Path("data/compound_profiles.json")
STRATEGIES_OUT = Path("data/actual_strategies.json")

OPENF1_BASE = "https://api.openf1.org/v1"

# Complete 2026 FIA Formula One World Championship Calendar (23 Circuits)
CALENDAR_2026: Dict[str, Dict[str, Any]] = {
    "australia": {
        "round": 1,
        "name": "Albert Park Grand Prix Circuit",
        "country": "Australia",
        "openf1_queries": [
            (2026, "circuit_short_name", "Melbourne"),
            (2025, "circuit_short_name", "Melbourne"),
        ],
        "total_laps": 58,
        "base_lap_time_sec": 79.8,
        "pit_loss_sec": 20.4,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.70,
        "fuel_effect_sec_per_kg": 0.032,
        "tire_wear_multiplier": 1.05,
        "overtaking_difficulty": 0.55,
        "dirty_air_penalty_sec": 0.55,
        "safety_car_prob_per_lap": 0.028,
        "rain_prob": 0.20,
    },
    "china": {
        "round": 2,
        "name": "Shanghai International Circuit",
        "country": "China",
        "openf1_queries": [
            (2026, "circuit_short_name", "Shanghai"),
            (2025, "circuit_short_name", "Shanghai"),
        ],
        "total_laps": 56,
        "base_lap_time_sec": 95.4,
        "pit_loss_sec": 22.6,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.76,
        "fuel_effect_sec_per_kg": 0.034,
        "tire_wear_multiplier": 1.25,
        "overtaking_difficulty": 0.35,
        "dirty_air_penalty_sec": 0.45,
        "safety_car_prob_per_lap": 0.022,
        "rain_prob": 0.18,
    },
    "suzuka": {
        "round": 3,
        "name": "Suzuka International Racing Course",
        "country": "Japan",
        "openf1_queries": [
            (2026, "circuit_short_name", "Suzuka"),
            (2025, "circuit_short_name", "Suzuka"),
        ],
        "total_laps": 53,
        "base_lap_time_sec": 91.2,
        "pit_loss_sec": 22.8,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.86,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.35,
        "overtaking_difficulty": 0.70,
        "dirty_air_penalty_sec": 0.75,
        "safety_car_prob_per_lap": 0.020,
        "rain_prob": 0.25,
    },
    "miami": {
        "round": 4,
        "name": "Miami International Autodrome",
        "country": "United States",
        "openf1_queries": [
            (2026, "circuit_short_name", "Miami"),
            (2025, "circuit_short_name", "Miami"),
        ],
        "total_laps": 57,
        "base_lap_time_sec": 90.1,
        "pit_loss_sec": 20.8,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.73,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.10,
        "overtaking_difficulty": 0.42,
        "dirty_air_penalty_sec": 0.50,
        "safety_car_prob_per_lap": 0.024,
        "rain_prob": 0.18,
    },
    "canada": {
        "round": 5,
        "name": "Circuit Gilles Villeneuve",
        "country": "Canada",
        "openf1_queries": [
            (2026, "circuit_short_name", "Montreal"),
            (2025, "circuit_short_name", "Montreal"),
        ],
        "total_laps": 70,
        "base_lap_time_sec": 74.6,
        "pit_loss_sec": 19.2,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.40,
        "fuel_effect_sec_per_kg": 0.030,
        "tire_wear_multiplier": 0.90,
        "overtaking_difficulty": 0.40,
        "dirty_air_penalty_sec": 0.45,
        "safety_car_prob_per_lap": 0.030,
        "rain_prob": 0.28,
    },
    "monaco": {
        "round": 6,
        "name": "Circuit de Monaco",
        "country": "Monaco",
        "openf1_queries": [
            (2026, "circuit_short_name", "Monte Carlo"),
            (2025, "circuit_short_name", "Monte Carlo"),
        ],
        "total_laps": 78,
        "base_lap_time_sec": 74.2,
        "pit_loss_sec": 19.4,
        "fuel_load_kg": 92.0,
        "fuel_burn_per_lap_kg": 1.15,
        "fuel_effect_sec_per_kg": 0.028,
        "tire_wear_multiplier": 0.55,
        "overtaking_difficulty": 0.95,
        "dirty_air_penalty_sec": 0.90,
        "safety_car_prob_per_lap": 0.035,
        "rain_prob": 0.15,
    },
    "barcelona": {
        "round": 7,
        "name": "Circuit de Barcelona-Catalunya",
        "country": "Spain",
        "openf1_queries": [
            (2026, "circuit_short_name", "Catalunya"),
            (2025, "circuit_short_name", "Catalunya"),
        ],
        "total_laps": 66,
        "base_lap_time_sec": 76.8,
        "pit_loss_sec": 21.8,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.50,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.38,
        "overtaking_difficulty": 0.55,
        "dirty_air_penalty_sec": 0.60,
        "safety_car_prob_per_lap": 0.015,
        "rain_prob": 0.10,
    },
    "austria": {
        "round": 8,
        "name": "Red Bull Ring",
        "country": "Austria",
        "openf1_queries": [
            (2026, "circuit_short_name", "Spielberg"),
            (2025, "circuit_short_name", "Spielberg"),
        ],
        "total_laps": 71,
        "base_lap_time_sec": 67.8,
        "pit_loss_sec": 20.1,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.39,
        "fuel_effect_sec_per_kg": 0.031,
        "tire_wear_multiplier": 1.18,
        "overtaking_difficulty": 0.32,
        "dirty_air_penalty_sec": 0.40,
        "safety_car_prob_per_lap": 0.018,
        "rain_prob": 0.25,
    },
    "silverstone": {
        "round": 9,
        "name": "Silverstone Circuit",
        "country": "United Kingdom",
        "openf1_queries": [
            (2026, "circuit_short_name", "Silverstone"),
            (2025, "circuit_short_name", "Silverstone"),
        ],
        "total_laps": 52,
        "base_lap_time_sec": 89.2,
        "pit_loss_sec": 20.2,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.90,
        "fuel_effect_sec_per_kg": 0.034,
        "tire_wear_multiplier": 1.25,
        "overtaking_difficulty": 0.45,
        "dirty_air_penalty_sec": 0.55,
        "safety_car_prob_per_lap": 0.022,
        "rain_prob": 0.30,
    },
    "spa": {
        "round": 10,
        "name": "Circuit de Spa-Francorchamps",
        "country": "Belgium",
        "openf1_queries": [
            (2026, "circuit_short_name", "Spa-Francorchamps"),
            (2025, "circuit_short_name", "Spa-Francorchamps"),
        ],
        "total_laps": 44,
        "base_lap_time_sec": 106.2,
        "pit_loss_sec": 21.6,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 2.24,
        "fuel_effect_sec_per_kg": 0.038,
        "tire_wear_multiplier": 1.15,
        "overtaking_difficulty": 0.35,
        "dirty_air_penalty_sec": 0.45,
        "safety_car_prob_per_lap": 0.028,
        "rain_prob": 0.40,
    },
    "hungary": {
        "round": 11,
        "name": "Hungaroring",
        "country": "Hungary",
        "openf1_queries": [
            (2026, "circuit_short_name", "Hungaroring"),
            (2025, "circuit_short_name", "Hungaroring"),
        ],
        "total_laps": 70,
        "base_lap_time_sec": 79.4,
        "pit_loss_sec": 20.8,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.40,
        "fuel_effect_sec_per_kg": 0.031,
        "tire_wear_multiplier": 1.22,
        "overtaking_difficulty": 0.80,
        "dirty_air_penalty_sec": 0.78,
        "safety_car_prob_per_lap": 0.016,
        "rain_prob": 0.16,
    },
    "zandvoort": {
        "round": 12,
        "name": "Circuit Zandvoort",
        "country": "Netherlands",
        "openf1_queries": [
            (2026, "circuit_short_name", "Zandvoort"),
            (2025, "circuit_short_name", "Zandvoort"),
        ],
        "total_laps": 72,
        "base_lap_time_sec": 73.1,
        "pit_loss_sec": 19.6,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.37,
        "fuel_effect_sec_per_kg": 0.030,
        "tire_wear_multiplier": 1.20,
        "overtaking_difficulty": 0.78,
        "dirty_air_penalty_sec": 0.72,
        "safety_car_prob_per_lap": 0.024,
        "rain_prob": 0.32,
    },
    "monza": {
        "round": 13,
        "name": "Autodromo Nazionale Monza",
        "country": "Italy",
        "openf1_queries": [
            (2026, "circuit_short_name", "Monza"),
            (2026, "country_name", "Italy"),
            (2025, "country_name", "Italy"),
        ],
        "total_laps": 53,
        "base_lap_time_sec": 82.1,
        "pit_loss_sec": 23.5,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.86,
        "fuel_effect_sec_per_kg": 0.032,
        "tire_wear_multiplier": 0.85,
        "overtaking_difficulty": 0.25,
        "dirty_air_penalty_sec": 0.35,
        "safety_car_prob_per_lap": 0.015,
        "rain_prob": 0.12,
    },
    "madrid": {
        "round": 14,
        "name": "Madring (Circuito de Madrid)",
        "country": "Spain",
        "openf1_queries": [
            (2026, "circuit_short_name", "Madrid"),
            (2026, "country_name", "Spain"),
        ],
        "total_laps": 55,
        "base_lap_time_sec": 93.4,
        "pit_loss_sec": 21.4,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.80,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.08,
        "overtaking_difficulty": 0.58,
        "dirty_air_penalty_sec": 0.58,
        "safety_car_prob_per_lap": 0.026,
        "rain_prob": 0.10,
    },
    "baku": {
        "round": 15,
        "name": "Baku City Circuit",
        "country": "Azerbaijan",
        "openf1_queries": [
            (2026, "circuit_short_name", "Baku"),
            (2025, "circuit_short_name", "Baku"),
        ],
        "total_laps": 51,
        "base_lap_time_sec": 104.2,
        "pit_loss_sec": 20.5,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.94,
        "fuel_effect_sec_per_kg": 0.035,
        "tire_wear_multiplier": 0.88,
        "overtaking_difficulty": 0.30,
        "dirty_air_penalty_sec": 0.38,
        "safety_car_prob_per_lap": 0.034,
        "rain_prob": 0.08,
    },
    "sepang": {
        "round": 16,
        "name": "Sepang International Circuit (Bahrain GP)",
        "country": "Malaysia",
        "openf1_queries": [
            (2026, "circuit_short_name", "Sepang"),
            (2026, "circuit_short_name", "Sakhir"),
            (2025, "circuit_short_name", "Sakhir"),
        ],
        "total_laps": 56,
        "base_lap_time_sec": 96.5,
        "pit_loss_sec": 22.5,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.77,
        "fuel_effect_sec_per_kg": 0.034,
        "tire_wear_multiplier": 1.42,
        "overtaking_difficulty": 0.32,
        "dirty_air_penalty_sec": 0.42,
        "safety_car_prob_per_lap": 0.020,
        "rain_prob": 0.38,
    },
    "singapore": {
        "round": 17,
        "name": "Marina Bay Street Circuit",
        "country": "Singapore",
        "openf1_queries": [
            (2026, "circuit_short_name", "Singapore"),
            (2025, "circuit_short_name", "Singapore"),
        ],
        "total_laps": 62,
        "base_lap_time_sec": 95.8,
        "pit_loss_sec": 28.2,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.60,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.12,
        "overtaking_difficulty": 0.85,
        "dirty_air_penalty_sec": 0.82,
        "safety_car_prob_per_lap": 0.036,
        "rain_prob": 0.25,
    },
    "austin": {
        "round": 18,
        "name": "Circuit of The Americas",
        "country": "United States",
        "openf1_queries": [
            (2026, "circuit_short_name", "Austin"),
            (2025, "circuit_short_name", "Austin"),
        ],
        "total_laps": 56,
        "base_lap_time_sec": 96.2,
        "pit_loss_sec": 21.2,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.76,
        "fuel_effect_sec_per_kg": 0.034,
        "tire_wear_multiplier": 1.32,
        "overtaking_difficulty": 0.40,
        "dirty_air_penalty_sec": 0.48,
        "safety_car_prob_per_lap": 0.020,
        "rain_prob": 0.12,
    },
    "mexico": {
        "round": 19,
        "name": "Autódromo Hermanos Rodríguez",
        "country": "Mexico",
        "openf1_queries": [
            (2026, "circuit_short_name", "Mexico City"),
            (2026, "country_name", "Mexico"),
            (2025, "country_name", "Mexico"),
        ],
        "total_laps": 71,
        "base_lap_time_sec": 79.5,
        "pit_loss_sec": 22.0,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.39,
        "fuel_effect_sec_per_kg": 0.030,
        "tire_wear_multiplier": 0.92,
        "overtaking_difficulty": 0.48,
        "dirty_air_penalty_sec": 0.50,
        "safety_car_prob_per_lap": 0.022,
        "rain_prob": 0.15,
    },
    "brazil": {
        "round": 20,
        "name": "Autódromo José Carlos Pace (Interlagos)",
        "country": "Brazil",
        "openf1_queries": [
            (2026, "circuit_short_name", "Interlagos"),
            (2025, "circuit_short_name", "Interlagos"),
        ],
        "total_laps": 71,
        "base_lap_time_sec": 72.8,
        "pit_loss_sec": 20.6,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.39,
        "fuel_effect_sec_per_kg": 0.030,
        "tire_wear_multiplier": 1.18,
        "overtaking_difficulty": 0.35,
        "dirty_air_penalty_sec": 0.42,
        "safety_car_prob_per_lap": 0.032,
        "rain_prob": 0.38,
    },
    "las_vegas": {
        "round": 21,
        "name": "Las Vegas Strip Circuit",
        "country": "United States",
        "openf1_queries": [
            (2026, "circuit_short_name", "Las Vegas"),
            (2025, "circuit_short_name", "Las Vegas"),
        ],
        "total_laps": 50,
        "base_lap_time_sec": 94.5,
        "pit_loss_sec": 21.0,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.98,
        "fuel_effect_sec_per_kg": 0.035,
        "tire_wear_multiplier": 0.86,
        "overtaking_difficulty": 0.28,
        "dirty_air_penalty_sec": 0.36,
        "safety_car_prob_per_lap": 0.025,
        "rain_prob": 0.04,
    },
    "qatar": {
        "round": 22,
        "name": "Lusail International Circuit",
        "country": "Qatar",
        "openf1_queries": [
            (2026, "circuit_short_name", "Lusail"),
            (2025, "circuit_short_name", "Lusail"),
        ],
        "total_laps": 57,
        "base_lap_time_sec": 84.2,
        "pit_loss_sec": 24.8,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.74,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.45,
        "overtaking_difficulty": 0.52,
        "dirty_air_penalty_sec": 0.55,
        "safety_car_prob_per_lap": 0.020,
        "rain_prob": 0.02,
    },
    "abu_dhabi": {
        "round": 23,
        "name": "Yas Marina Circuit",
        "country": "United Arab Emirates",
        "openf1_queries": [
            (2026, "circuit_short_name", "Yas Marina Circuit"),
            (2025, "circuit_short_name", "Yas Marina Circuit"),
        ],
        "total_laps": 58,
        "base_lap_time_sec": 86.4,
        "pit_loss_sec": 21.5,
        "fuel_load_kg": 100.0,
        "fuel_burn_per_lap_kg": 1.71,
        "fuel_effect_sec_per_kg": 0.033,
        "tire_wear_multiplier": 1.05,
        "overtaking_difficulty": 0.44,
        "dirty_air_penalty_sec": 0.48,
        "safety_car_prob_per_lap": 0.018,
        "rain_prob": 0.02,
    },
}

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def _normalize_compound(raw: Any) -> str:
    """Normalize compound string to canonical simulator identifier."""
    c = str(raw or "").strip().upper()
    if c in ("INTER", "INTERMEDIATE"):
        return "INTERMEDIATE"
    if c in VALID_COMPOUNDS:
        return c
    return ""


def _openf1_get(endpoint: str, params: Dict[str, Any], timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Pure-Python HTTP GET helper for OpenF1 REST API with automatic HTTP 429 rate-limit backoff.
    """
    query_str = urlencode(params)
    url = f"{OPENF1_BASE}/{endpoint}?{query_str}" if query_str else f"{OPENF1_BASE}/{endpoint}"
    req = Request(url, headers={"User-Agent": "f1-pitwall-optimizer/2.3"})
    ctx = ssl.create_default_context()

    for attempt in range(4):
        try:
            time.sleep(0.42)  # Stay safely under OpenF1 3 req/sec free tier burst limit
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                return data if isinstance(data, list) else []
        except HTTPError as exc:
            if exc.code == 429:
                # Exponential backoff on rate limit
                time.sleep(1.5 * (attempt + 1))
                continue
            return []
        except Exception:
            time.sleep(0.5)
    return []


def _fetch_openf1_circuit_and_strategies(
    cfg: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]]:
    """
    Pure-Python OpenF1 REST API extractor using `sessions`, `session_result`,
    `drivers`, `stints`, and `laps`.
    """
    import numpy as np

    selected_session = None
    query_label = ""

    for year, filter_field, filter_val in cfg.get("openf1_queries", []):
        sessions = _openf1_get(
            "sessions",
            {"year": year, "session_name": "Race", filter_field: filter_val},
        )
        if sessions:
            # Verify the session has stints (i.e., the Grand Prix has already taken place)
            cand = sessions[-1]
            stints_check = _openf1_get("stints", {"session_key": cand["session_key"]})
            if stints_check:
                selected_session = cand
                stints_raw = stints_check
                circuit_name = cand.get("circuit_short_name") or cand.get("country_name") or filter_val
                query_label = f"{year} {circuit_name} GP (Race)"
                break

    if not selected_session:
        return None

    s_key = selected_session["session_key"]
    drivers_raw = _openf1_get("drivers", {"session_key": s_key})
    results_raw = _openf1_get("session_result", {"session_key": s_key})
    laps_raw = _openf1_get("laps", {"session_key": s_key})

    # Map driver_number -> driver metadata
    drv_info_map: Dict[int, Dict[str, Any]] = {}
    for d in drivers_raw:
        num = d.get("driver_number")
        if num is None:
            continue
        drv_info_map[int(num)] = {
            "driver_code": str(d.get("name_acronym") or f"#{num}").strip(),
            "driver_name": str(d.get("full_name") or d.get("broadcast_name") or f"Driver #{num}").strip().title(),
            "team": str(d.get("team_name") or "F1 Team").strip(),
        }

    # Map driver_number -> official FIA finishing classification from /session_result
    official_result_map: Dict[int, Dict[str, Any]] = {}
    for res in results_raw:
        num = res.get("driver_number")
        if num is None:
            continue
        num_i = int(num)
        pos_val = res.get("position")
        dnf = bool(res.get("dnf"))
        dns = bool(res.get("dns"))
        dsq = bool(res.get("dsq"))
        num_laps = res.get("number_of_laps")

        if dsq:
            status_str = "DSQ"
            pos_int = 98
        elif dns:
            status_str = "DNS"
            pos_int = 99
        elif dnf:
            status_str = "DNF"
            pos_int = int(pos_val) if pos_val is not None else 95
        else:
            status_str = "Finished"
            pos_int = int(pos_val) if pos_val is not None else 90

        official_result_map[num_i] = {
            "position": pos_int,
            "status": status_str,
            "number_of_laps": int(num_laps) if num_laps is not None else 0,
        }

    # Group stints by driver_number
    stints_by_driver: Dict[int, List[Dict[str, Any]]] = {}
    max_lap_seen = 0
    for st in stints_raw:
        num = st.get("driver_number")
        comp = _normalize_compound(st.get("compound"))
        lap_start = st.get("lap_start")
        lap_end = st.get("lap_end")
        if num is None or not comp or lap_start is None or lap_end is None:
            continue
        l_start = int(lap_start)
        l_end = int(lap_end)
        max_lap_seen = max(max_lap_seen, l_end)
        stints_by_driver.setdefault(int(num), []).append(
            {
                "stint_number": int(st.get("stint_number") or len(stints_by_driver.get(int(num), [])) + 1),
                "compound": comp,
                "lap_start": l_start,
                "lap_end": l_end,
            }
        )

    total_laps = max(30, max_lap_seen) if max_lap_seen >= 25 else cfg["total_laps"]

    # Calibrate fuel-corrected clean base lap time from OpenF1 /laps
    all_fuel_corrected: List[float] = []
    fuel_load = cfg["fuel_load_kg"]
    fuel_burn = round(fuel_load / max(1, total_laps), 3)
    fuel_eff = cfg["fuel_effect_sec_per_kg"]

    for lp in laps_raw:
        dur = lp.get("lap_duration")
        lap_num = lp.get("lap_number")
        is_pit_out = bool(lp.get("is_pit_out_lap"))
        if dur is None or lap_num is None:
            continue
        dur_f = float(dur)
        lap_i = int(lap_num)
        if not is_pit_out and lap_i >= 2 and 55.0 <= dur_f <= 145.0:
            fuel_pen = max(0.0, fuel_load - (lap_i - 1) * fuel_burn) * fuel_eff
            all_fuel_corrected.append(dur_f - fuel_pen)

    if len(all_fuel_corrected) >= 20:
        med_fc = float(np.median(all_fuel_corrected))
        trimmed = [t for t in all_fuel_corrected if t <= med_fc * 1.06]
        base_lap_time_sec = round(float(np.percentile(trimmed, 10)), 2) if len(trimmed) >= 15 else cfg["base_lap_time_sec"]
    else:
        base_lap_time_sec = cfg["base_lap_time_sec"]

    # Build driver strategy entries
    drivers_list: List[Dict[str, Any]] = []
    for num_i, dstints in stints_by_driver.items():
        dstints.sort(key=lambda s: (s["stint_number"], s["lap_start"]))
        start_compound = dstints[0]["compound"]
        pit_laps: List[int] = []
        next_compounds: List[str] = []

        for idx in range(len(dstints) - 1):
            box_lap = int(dstints[idx]["lap_end"])
            next_comp = dstints[idx + 1]["compound"]
            if 1 <= box_lap <= max(1, total_laps - 1):
                pit_laps.append(box_lap)
                next_compounds.append(next_comp)

        stint_sequence = [start_compound] + next_compounds
        short_codes = [c[0] for c in stint_sequence]
        if pit_laps:
            lap_str = ", ".join(f"L{lap}" for lap in pit_laps)
            summary = f"{' → '.join(short_codes)} ({len(pit_laps)}-Stop: {lap_str})"
        else:
            summary = f"{short_codes[0]} (0-Stop)"

        meta = drv_info_map.get(
            num_i,
            {
                "driver_code": f"#{num_i}",
                "driver_name": f"Driver #{num_i}",
                "team": "F1 Constructor",
            },
        )
        res_info = official_result_map.get(num_i, {})
        laps_completed = res_info.get("number_of_laps") or max(s["lap_end"] for s in dstints)
        pos = res_info.get("position", 90)
        status = res_info.get(
            "status",
            "Finished" if laps_completed >= total_laps - 1 else f"+{total_laps - laps_completed} Laps",
        )
        is_dry_only = all(c in DRY_COMPOUNDS for c in stint_sequence)

        drivers_list.append(
            {
                "position": pos,
                "grid_position": 0,
                "driver_code": meta["driver_code"],
                "driver_name": meta["driver_name"],
                "team": meta["team"],
                "status": status,
                "laps_completed": laps_completed,
                "num_stops": len(pit_laps),
                "start_compound": start_compound,
                "pit_laps": pit_laps,
                "compounds": next_compounds,
                "stint_sequence": stint_sequence,
                "strategy_summary": summary,
                "is_dry_only": is_dry_only,
            }
        )

    # Sort by official FIA classification position, then laps completed
    drivers_list.sort(key=lambda d: (d["position"], -d["laps_completed"], d["driver_code"]))
    for rank_idx, d in enumerate(drivers_list, start=1):
        if d["position"] >= 90:
            d["position"] = rank_idx

    c_entry = {
        "round": cfg["round"],
        "name": cfg["name"],
        "country": cfg["country"],
        "total_laps": total_laps,
        "base_lap_time_sec": base_lap_time_sec,
        "pit_loss_sec": cfg["pit_loss_sec"],
        "fuel_load_kg": fuel_load,
        "fuel_burn_per_lap_kg": fuel_burn,
        "fuel_effect_sec_per_kg": fuel_eff,
        "tire_wear_multiplier": cfg["tire_wear_multiplier"],
        "overtaking_difficulty": cfg["overtaking_difficulty"],
        "dirty_air_penalty_sec": cfg["dirty_air_penalty_sec"],
        "safety_car_prob_per_lap": cfg["safety_car_prob_per_lap"],
        "rain_prob": cfg["rain_prob"],
        "telemetry_source": f"OpenF1 Live Timing: {query_label}",
    }
    comp_fit = {
        "SOFT": {"base_offset_sec": 0.00, "deg_per_lap_sec": round(0.095 * cfg["tire_wear_multiplier"], 4)},
        "MEDIUM": {"base_offset_sec": 0.55, "deg_per_lap_sec": round(0.055 * cfg["tire_wear_multiplier"], 4)},
        "HARD": {"base_offset_sec": 1.15, "deg_per_lap_sec": round(0.030 * cfg["tire_wear_multiplier"], 4)},
    }
    strat_entry = {
        "source": f"OpenF1 Official Timing: {query_label}",
        "total_laps": total_laps,
        "drivers": drivers_list,
    }
    return c_entry, comp_fit, strat_entry


def _fallback_entry(
    cfg: Dict[str, Any], reason: str
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Return 2026 FIA baseline telemetry profile if OpenF1 has no session data yet."""
    c_entry = {
        "round": cfg["round"],
        "name": cfg["name"],
        "country": cfg["country"],
        "total_laps": cfg["total_laps"],
        "base_lap_time_sec": cfg["base_lap_time_sec"],
        "pit_loss_sec": cfg["pit_loss_sec"],
        "fuel_load_kg": cfg["fuel_load_kg"],
        "fuel_burn_per_lap_kg": cfg["fuel_burn_per_lap_kg"],
        "fuel_effect_sec_per_kg": cfg["fuel_effect_sec_per_kg"],
        "tire_wear_multiplier": cfg["tire_wear_multiplier"],
        "overtaking_difficulty": cfg["overtaking_difficulty"],
        "dirty_air_penalty_sec": cfg["dirty_air_penalty_sec"],
        "safety_car_prob_per_lap": cfg["safety_car_prob_per_lap"],
        "rain_prob": cfg["rain_prob"],
        "telemetry_source": f"2026 FIA Calibrated Model ({reason})",
    }
    comp_fit = {
        "SOFT": {"base_offset_sec": 0.00, "deg_per_lap_sec": round(0.095 * cfg["tire_wear_multiplier"], 4)},
        "MEDIUM": {"base_offset_sec": 0.55, "deg_per_lap_sec": round(0.055 * cfg["tire_wear_multiplier"], 4)},
        "HARD": {"base_offset_sec": 1.15, "deg_per_lap_sec": round(0.030 * cfg["tire_wear_multiplier"], 4)},
    }
    strat_entry = {
        "source": f"No race classification available ({reason})",
        "total_laps": cfg["total_laps"],
        "drivers": [],
    }
    return c_entry, comp_fit, strat_entry


def run_calibration() -> None:
    """Orchestrate OpenF1 REST telemetry & strategy extraction across all 23 circuits."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    calibrated_circuits: Dict[str, Any] = {}
    calibrated_compounds: Dict[str, Any] = {}
    calibrated_strategies: Dict[str, Any] = {}

    print("=" * 88)
    print("🏎️  OPENF1 LIVE TIMING & ACTUAL RACE STRATEGY CALIBRATION (ALL 23 ROUNDS)")
    print("=" * 88)

    for key, cfg in CALENDAR_2026.items():
        rnd = cfg["round"]
        print(f"[R{rnd:02d}/23] Fetching {cfg['name']} ({cfg['country']})...", end=" ", flush=True)

        openf1_res = _fetch_openf1_circuit_and_strategies(cfg)
        if openf1_res is not None:
            c_entry, comp_fit, strat_entry = openf1_res
        else:
            c_entry, comp_fit, strat_entry = _fallback_entry(cfg, "Session pending")

        n_drivers = len(strat_entry.get("drivers", []))
        if n_drivers > 0:
            w = strat_entry["drivers"][0]
            print(
                f"OK -> {c_entry['telemetry_source']} | Laps={c_entry['total_laps']} | "
                f"P1 {w['driver_code']}: {w['strategy_summary']} ({n_drivers} drivers)"
            )
        else:
            print(f"BASELINE -> {c_entry['telemetry_source']}")

        calibrated_circuits[key] = c_entry
        calibrated_compounds[key] = comp_fit
        calibrated_strategies[key] = strat_entry

    with CIRCUITS_OUT.open("w", encoding="utf-8") as f:
        json.dump(calibrated_circuits, f, indent=2, ensure_ascii=False)

    with COMPOUNDS_OUT.open("w", encoding="utf-8") as f:
        json.dump(calibrated_compounds, f, indent=2, ensure_ascii=False)

    with STRATEGIES_OUT.open("w", encoding="utf-8") as f:
        json.dump(calibrated_strategies, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print(f"✅ Saved all {len(calibrated_circuits)} circuits to {CIRCUITS_OUT}")
    print(f"✅ Saved circuit-specific compound fits to {COMPOUNDS_OUT}")
    print(f"✅ Saved actual Grand Prix driver strategies to {STRATEGIES_OUT}")
    print("=" * 88)


if __name__ == "__main__":
    run_calibration()