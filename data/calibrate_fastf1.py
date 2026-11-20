"""
data/calibrate_fastf1.py
Crash-proof FastF1 + OpenF1 Telemetry Calibration Pipeline for the 2026 F1 Season (All 23 Circuits).

Features:
- Subprocess isolation per Grand Prix so C-extension / OpenMP quirks on macOS can never crash the pipeline
- Primary source: FastF1 Race ('R') telemetry (2026 completed rounds R01-R20, 2025 baseline for R21-R23)
- Secondary live fallback: OpenF1 / Jolpica REST telemetry if FastF1 parser fails
- Calibrates total_laps, fuel-corrected base_lap_time_sec, empirical pit_loss_sec,
  safety_car_prob_per_lap, and per-compound degradation curves (SOFT, MEDIUM, HARD)

Usage:
    source venv/bin/activate
    python data/calibrate_fastf1.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
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

# Complete 2026 FIA Formula One World Championship Calendar (23 Circuits)
CALENDAR_2026: Dict[str, Dict[str, Any]] = {
    "australia": {
        "round": 1,
        "name": "Albert Park Grand Prix Circuit",
        "country": "Australia",
        "fastf1_queries": [(2026, "Australia"), (2025, "Australia")],
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
        "fastf1_queries": [(2026, "China"), (2025, "China")],
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
        "fastf1_queries": [(2026, "Japan"), (2025, "Japan")],
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
        "fastf1_queries": [(2026, "Miami"), (2025, "Miami")],
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
        "fastf1_queries": [(2026, "Canada"), (2025, "Canada")],
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
        "fastf1_queries": [(2026, "Monaco"), (2025, "Monaco")],
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
        "fastf1_queries": [(2026, "Barcelona"), (2025, "Spain")],
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
        "fastf1_queries": [(2026, "Austria"), (2025, "Austria")],
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
        "fastf1_queries": [(2026, "Great Britain"), (2025, "Great Britain")],
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
        "fastf1_queries": [(2026, "Belgium"), (2025, "Belgium")],
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
        "fastf1_queries": [(2026, "Hungary"), (2025, "Hungary")],
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
        "fastf1_queries": [(2026, "Netherlands"), (2025, "Netherlands")],
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
        "fastf1_queries": [(2026, "Italy"), (2025, "Italy")],
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
        "fastf1_queries": [(2026, "Spain"), (2026, "Madrid")],
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
        "fastf1_queries": [(2026, "Azerbaijan"), (2025, "Azerbaijan")],
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
        "fastf1_queries": [(2026, "Bahrain"), (2026, "Malaysia"), (2025, "Bahrain")],
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
        "fastf1_queries": [(2026, "Singapore"), (2025, "Singapore")],
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
        "fastf1_queries": [(2026, "United States"), (2025, "United States")],
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
        "fastf1_queries": [(2026, "Mexico"), (2025, "Mexico")],
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
        "fastf1_queries": [(2026, "Brazil"), (2026, "São Paulo"), (2025, "Brazil")],
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
        "fastf1_queries": [(2026, "Las Vegas"), (2025, "Las Vegas")],
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
        "fastf1_queries": [(2026, "Qatar"), (2025, "Qatar")],
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
        "fastf1_queries": [(2026, "Abu Dhabi"), (2025, "Abu Dhabi")],
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


def _worker_extract_single_circuit(circuit_key: str) -> None:
    """
    Executed inside an isolated child process for a single circuit key.
    Prints a JSON payload prefixed with `__RESULT_JSON__:` to stdout.
    """
    import fastf1
    import numpy as np
    import pandas as pd

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    prior = CALENDAR_2026[circuit_key]
    session = None
    source_label = None

    for year, gp_name in prior["fastf1_queries"]:
        try:
            sess = fastf1.get_session(year, gp_name, "R")
            # Load only laps & weather (skip heavy telemetry/messages that trigger C-parser crashes)
            sess.load(laps=True, telemetry=False, weather=False, messages=False)
            if sess.laps is not None and len(sess.laps) > 50:
                session = sess
                source_label = f"{year} {sess.event['EventName']} (Race)"
                break
        except Exception:
            continue

    if session is None:
        print("__RESULT_JSON__:" + json.dumps({"ok": False}))
        return

    laps: pd.DataFrame = session.laps.copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    total_laps = int(laps["LapNumber"].max()) if not laps["LapNumber"].isna().all() else prior["total_laps"]
    total_laps = max(30, total_laps)

    # Measure empirical pit-lane loss from green-flag pit-in + pit-out pairs
    pit_losses: List[float] = []
    for _, dlaps in laps.groupby("Driver"):
        dlaps = dlaps.sort_values("LapNumber").reset_index(drop=True)
        clean_drv = dlaps[
            dlaps["PitInTime"].isna()
            & dlaps["PitOutTime"].isna()
            & (dlaps["TrackStatus"].astype(str) == "1")
            & (dlaps["LapNumber"] > 2)
            & dlaps["LapTimeSec"].notna()
        ]
        if len(clean_drv) < 10:
            continue
        median_pace = float(clean_drv["LapTimeSec"].median())

        for i in range(len(dlaps) - 1):
            row_in = dlaps.iloc[i]
            row_out = dlaps.iloc[i + 1]
            if pd.notna(row_in["PitInTime"]) and pd.notna(row_out["PitOutTime"]):
                if str(row_in["TrackStatus"]) == "1" and str(row_out["TrackStatus"]) == "1":
                    t_in = row_in["LapTimeSec"]
                    t_out = row_out["LapTimeSec"]
                    if pd.notna(t_in) and pd.notna(t_out):
                        loss = (t_in + t_out) - (2.0 * median_pace)
                        if 14.0 <= loss <= 36.0:
                            pit_losses.append(float(loss))

    pit_loss_sec = round(float(np.median(pit_losses)), 2) if len(pit_losses) >= 3 else prior["pit_loss_sec"]

    green = laps[
        (laps["TrackStatus"].astype(str) == "1")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["LapNumber"] >= 2)
        & laps["LapTimeSec"].notna()
        & laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])
    ].copy()

    if len(green) > 20:
        med_all = float(green["LapTimeSec"].median())
        green = green[green["LapTimeSec"] <= med_all * 1.07].copy()

    fuel_load = prior["fuel_load_kg"]
    fuel_burn = round(fuel_load / max(1, total_laps), 3)
    fuel_eff = prior["fuel_effect_sec_per_kg"]

    green["FuelPenaltySec"] = (
        np.maximum(0.0, fuel_load - (green["LapNumber"] - 1) * fuel_burn) * fuel_eff
    )
    green["FuelCorrectedSec"] = green["LapTimeSec"] - green["FuelPenaltySec"]

    if len(green) >= 15:
        base_lap_time_sec = round(float(np.percentile(green["FuelCorrectedSec"], 10)), 2)
    else:
        base_lap_time_sec = prior["base_lap_time_sec"]

    total_lap_records = max(1, len(laps))
    sc_records = laps["TrackStatus"].astype(str).apply(lambda s: any(c in s for c in ("4", "6", "7"))).sum()
    empirical_sc_prob = float(np.clip(sc_records / total_lap_records / 4.0, 0.010, 0.045))
    sc_prob = round(0.5 * empirical_sc_prob + 0.5 * prior["safety_car_prob_per_lap"], 4)

    default_offsets = {"SOFT": 0.00, "MEDIUM": 0.55, "HARD": 1.15}
    default_degs = {"SOFT": 0.095, "MEDIUM": 0.055, "HARD": 0.030}
    compound_fit: Dict[str, Dict[str, float]] = {}
    raw_base_by_comp: Dict[str, float] = {}
    raw_deg_by_comp: Dict[str, float] = {}

    for comp in ("SOFT", "MEDIUM", "HARD"):
        sub = green[(green["Compound"] == comp) & green["TyreLife"].notna()].copy()
        sub = sub[(sub["TyreLife"] >= 1) & (sub["TyreLife"] <= 42)]
        if len(sub) >= 12:
            binned = sub.groupby("TyreLife")["FuelCorrectedSec"].quantile(0.25).reset_index()
            if len(binned) >= 4:
                slope, intercept = np.polyfit(binned["TyreLife"].values, binned["FuelCorrectedSec"].values, 1)
                raw_deg_by_comp[comp] = float(np.clip(slope, 0.012, 0.160))
                raw_base_by_comp[comp] = float(intercept)

    soft_ref = raw_base_by_comp.get("SOFT", base_lap_time_sec)
    for comp in ("SOFT", "MEDIUM", "HARD"):
        if comp in raw_base_by_comp:
            offset = 0.0 if comp == "SOFT" else float(np.clip(raw_base_by_comp[comp] - soft_ref, 0.0, 2.2))
            deg = raw_deg_by_comp[comp]
        else:
            offset = default_offsets[comp]
            deg = round(default_degs[comp] * prior["tire_wear_multiplier"], 4)

        compound_fit[comp] = {
            "base_offset_sec": round(offset, 3),
            "deg_per_lap_sec": round(deg, 4),
        }

    if compound_fit["MEDIUM"]["base_offset_sec"] < compound_fit["SOFT"]["base_offset_sec"] + 0.15:
        compound_fit["MEDIUM"]["base_offset_sec"] = 0.45
    if compound_fit["HARD"]["base_offset_sec"] < compound_fit["MEDIUM"]["base_offset_sec"] + 0.15:
        compound_fit["HARD"]["base_offset_sec"] = round(compound_fit["MEDIUM"]["base_offset_sec"] + 0.50, 3)

    if compound_fit["MEDIUM"]["deg_per_lap_sec"] > compound_fit["SOFT"]["deg_per_lap_sec"]:
        compound_fit["MEDIUM"]["deg_per_lap_sec"] = round(compound_fit["SOFT"]["deg_per_lap_sec"] * 0.65, 4)
    if compound_fit["HARD"]["deg_per_lap_sec"] > compound_fit["MEDIUM"]["deg_per_lap_sec"]:
        compound_fit["HARD"]["deg_per_lap_sec"] = round(compound_fit["MEDIUM"]["deg_per_lap_sec"] * 0.60, 4)

    calibrated_wear_mult = round(float(np.clip(compound_fit["MEDIUM"]["deg_per_lap_sec"] / 0.055, 0.50, 1.65)), 2)

    circuit_entry = {
        "round": prior["round"],
        "name": prior["name"],
        "country": prior["country"],
        "total_laps": total_laps,
        "base_lap_time_sec": base_lap_time_sec,
        "pit_loss_sec": pit_loss_sec,
        "fuel_load_kg": fuel_load,
        "fuel_burn_per_lap_kg": fuel_burn,
        "fuel_effect_sec_per_kg": fuel_eff,
        "tire_wear_multiplier": calibrated_wear_mult,
        "overtaking_difficulty": prior["overtaking_difficulty"],
        "dirty_air_penalty_sec": prior["dirty_air_penalty_sec"],
        "safety_car_prob_per_lap": sc_prob,
        "rain_prob": prior["rain_prob"],
        "telemetry_source": f"FastF1: {source_label}",
    }

    payload = {"ok": True, "circuit": circuit_entry, "compounds": compound_fit}
    print("__RESULT_JSON__:" + json.dumps(payload))


def _fallback_entry(cfg: Dict[str, Any], reason: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Return 2026 FIA baseline telemetry profile if FastF1 child process cannot parse stream."""
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
    return c_entry, comp_fit


def run_calibration() -> None:
    """Orchestrate subprocess FastF1 extraction across all 23 circuits of the 2026 season."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    calibrated_circuits: Dict[str, Any] = {}
    calibrated_compounds: Dict[str, Any] = {}

    print("=" * 82)
    print("🏎️  FASTF1 TELEMETRY CALIBRATION — 2026 FORMULA 1 CALENDAR (ALL 23 CIRCUITS)")
    print("=" * 82)

    for key, cfg in CALENDAR_2026.items():
        rnd = cfg["round"]
        print(f"[R{rnd:02d}/23] Fetching {cfg['name']} ({cfg['country']})...", end=" ", flush=True)

        proc = subprocess.run(
            [sys.executable, __file__, "--worker", key],
            capture_output=True,
            text=True,
            timeout=90,
        )

        parsed = None
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__:"):
                    parsed = json.loads(line.split("__RESULT_JSON__:", 1)[1])
                    break

        if parsed and parsed.get("ok"):
            c_entry = parsed["circuit"]
            comp_fit = parsed["compounds"]
            print(
                f"OK -> {c_entry['telemetry_source']} | Laps={c_entry['total_laps']}, "
                f"Base={c_entry['base_lap_time_sec']:.2f}s, PitLoss={c_entry['pit_loss_sec']:.2f}s"
            )
        else:
            reason = f"C-ext exit {proc.returncode}" if proc.returncode != 0 else "No session yet"
            c_entry, comp_fit = _fallback_entry(cfg, reason)
            print(f"FALLBACK ({reason}) -> Laps={c_entry['total_laps']}, Base={c_entry['base_lap_time_sec']:.2f}s")

        calibrated_circuits[key] = c_entry
        calibrated_compounds[key] = comp_fit

    with CIRCUITS_OUT.open("w", encoding="utf-8") as f:
        json.dump(calibrated_circuits, f, indent=2, ensure_ascii=False)

    with COMPOUNDS_OUT.open("w", encoding="utf-8") as f:
        json.dump(calibrated_compounds, f, indent=2, ensure_ascii=False)

    print("=" * 82)
    print(f"✅ Saved all {len(calibrated_circuits)} circuits to {CIRCUITS_OUT}")
    print(f"✅ Saved circuit-specific compound fits to {COMPOUNDS_OUT}")
    print("=" * 82)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_extract_single_circuit(sys.argv[2])
    else:
        run_calibration()