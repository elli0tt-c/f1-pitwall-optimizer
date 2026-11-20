"""
engine/physics.py
Lap-by-lap motorsport race physics engine calibrated with real FastF1 telemetry.

Models:
- Fuel mass burn and longitudinal/lateral lap-time sensitivity
- Circuit-specific empirical compound pace offsets & degradation slopes (from FastF1)
- Non-linear quadratic thermal/rubber performance cliff
- Wet-weather compound crossover physics (Dry vs. Intermediate vs. Full Wet)
- Wake aerodynamics (dirty-air downforce loss) and overtaking thresholds
- Pit-stop delta under Green Flag vs. Safety Car neutralization
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class TireCompound:
    """Physical and thermal characteristics of a racing tire compound."""
    name: str
    code: str
    color: str
    base_offset_sec: float      # Pace delta vs. fresh Soft on dry track (s)
    deg_per_lap_sec: float      # Nominal linear lap-time degradation rate (s/lap)
    cliff_lap: int              # Tire age (laps) where thermal/wear cliff begins
    cliff_severity: float       # Quadratic penalty coefficient past cliff_lap (s/lap^2)
    optimal_wetness_min: float  # Minimum track wetness [0.0 = bone dry, 1.0 = monsoon]
    optimal_wetness_max: float  # Maximum track wetness before aquaplaning


COMPOUNDS: Dict[str, TireCompound] = {
    "SOFT": TireCompound(
        name="Soft",
        code="S",
        color="#FF3333",
        base_offset_sec=0.00,
        deg_per_lap_sec=0.095,
        cliff_lap=14,
        cliff_severity=0.045,
        optimal_wetness_min=0.00,
        optimal_wetness_max=0.15,
    ),
    "MEDIUM": TireCompound(
        name="Medium",
        code="M",
        color="#FFD700",
        base_offset_sec=0.55,
        deg_per_lap_sec=0.055,
        cliff_lap=24,
        cliff_severity=0.030,
        optimal_wetness_min=0.00,
        optimal_wetness_max=0.15,
    ),
    "HARD": TireCompound(
        name="Hard",
        code="H",
        color="#F0F0F0",
        base_offset_sec=1.15,
        deg_per_lap_sec=0.030,
        cliff_lap=36,
        cliff_severity=0.022,
        optimal_wetness_min=0.00,
        optimal_wetness_max=0.12,
    ),
    "INTER": TireCompound(
        name="Intermediate",
        code="I",
        color="#39B54A",
        base_offset_sec=3.80,
        deg_per_lap_sec=0.045,
        cliff_lap=28,
        cliff_severity=0.025,
        optimal_wetness_min=0.15,
        optimal_wetness_max=0.65,
    ),
    "WET": TireCompound(
        name="Full Wet",
        code="W",
        color="#0080FF",
        base_offset_sec=7.20,
        deg_per_lap_sec=0.035,
        cliff_lap=35,
        cliff_severity=0.020,
        optimal_wetness_min=0.60,
        optimal_wetness_max=1.00,
    ),
}

DRY_COMPOUNDS: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")


@dataclass
class Circuit:
    """Circuit boundary conditions & empirical FastF1 compound fits."""
    key: str
    name: str
    country: str
    total_laps: int
    base_lap_time_sec: float
    pit_loss_sec: float
    fuel_load_kg: float
    fuel_burn_per_lap_kg: float
    fuel_effect_sec_per_kg: float
    tire_wear_multiplier: float
    overtaking_difficulty: float
    dirty_air_penalty_sec: float
    safety_car_prob_per_lap: float
    rain_prob: float
    round: int = 0
    telemetry_source: str = "FastF1 Calibrated"
    compound_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)


def load_circuits(
    json_path: str = "data/circuits.json",
    compounds_path: str = "data/compound_profiles.json",
) -> Dict[str, Circuit]:
    """
    Load all 2026 Grand Prix circuits and merge circuit-specific empirical
    compound degradation fits from `data/compound_profiles.json` if present.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    comp_profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    c_path = Path(compounds_path)
    if c_path.exists():
        with c_path.open("r", encoding="utf-8") as f:
            comp_profiles = json.load(f)

    circuits: Dict[str, Circuit] = {}
    for key, cfg in raw.items():
        circuits[key] = Circuit(
            key=key,
            name=cfg["name"],
            country=cfg["country"],
            total_laps=int(cfg["total_laps"]),
            base_lap_time_sec=float(cfg["base_lap_time_sec"]),
            pit_loss_sec=float(cfg["pit_loss_sec"]),
            fuel_load_kg=float(cfg["fuel_load_kg"]),
            fuel_burn_per_lap_kg=float(cfg["fuel_burn_per_lap_kg"]),
            fuel_effect_sec_per_kg=float(cfg["fuel_effect_sec_per_kg"]),
            tire_wear_multiplier=float(cfg["tire_wear_multiplier"]),
            overtaking_difficulty=float(cfg["overtaking_difficulty"]),
            dirty_air_penalty_sec=float(cfg["dirty_air_penalty_sec"]),
            safety_car_prob_per_lap=float(cfg["safety_car_prob_per_lap"]),
            rain_prob=float(cfg["rain_prob"]),
            round=int(cfg.get("round", 0)),
            telemetry_source=str(cfg.get("telemetry_source", "FastF1 Calibrated")),
            compound_overrides=comp_profiles.get(key, {}),
        )
    return circuits


def compute_fuel_penalty(circuit: Circuit, lap: int) -> float:
    """
    Calculate lap-time penalty (seconds) due to remaining fuel mass on `lap` (1-indexed).
    """
    laps_completed = max(0, lap - 1)
    remaining_fuel = max(0.0, circuit.fuel_load_kg - laps_completed * circuit.fuel_burn_per_lap_kg)
    return remaining_fuel * circuit.fuel_effect_sec_per_kg


def compute_tire_delta(
    compound_key: str,
    tire_age: int,
    circuit: Circuit,
    track_wetness: float = 0.0,
) -> float:
    """
    Calculate tire pace delta (seconds) relative to a fresh Soft on a dry track.

    Uses circuit-specific FastF1 empirical base offsets and linear degradation slopes
    when available, combined with quadratic thermal/wear cliff and wetness mismatch physics.
    """
    comp = COMPOUNDS[compound_key]
    effective_age = max(0, tire_age)

    override = circuit.compound_overrides.get(compound_key)
    if override is not None:
        base_offset = float(override.get("base_offset_sec", comp.base_offset_sec))
        # Empirical deg_per_lap_sec from FastF1 already incorporates track abrasiveness
        linear_deg = float(override.get("deg_per_lap_sec", comp.deg_per_lap_sec)) * effective_age
    else:
        base_offset = comp.base_offset_sec
        linear_deg = comp.deg_per_lap_sec * circuit.tire_wear_multiplier * effective_age

    # Non-linear performance cliff once tire exceeds thermal/wear life
    adjusted_cliff = max(5.0, comp.cliff_lap / max(0.5, circuit.tire_wear_multiplier ** 0.5))
    over_cliff = max(0.0, effective_age - adjusted_cliff)
    cliff_deg = comp.cliff_severity * (over_cliff ** 2)

    # Wetness mismatch penalty
    wetness_penalty = 0.0
    if track_wetness < comp.optimal_wetness_min:
        deficit = comp.optimal_wetness_min - track_wetness
        wetness_penalty = 18.0 * (deficit ** 1.5)
    elif track_wetness > comp.optimal_wetness_max:
        excess = track_wetness - comp.optimal_wetness_max
        wetness_penalty = 32.0 * (excess ** 1.6)

    return base_offset + linear_deg + cliff_deg + wetness_penalty


def compute_pit_loss(circuit: Circuit, is_safety_car: bool = False, noise_sec: float = 0.0) -> float:
    """
    Calculate pit-stop time loss (seconds).
    Under a Safety Car, field pace slows by ~38%, reducing relative pit loss by ~45%.
    """
    base_loss = circuit.pit_loss_sec * (0.55 if is_safety_car else 1.0)
    return max(8.0, base_loss + noise_sec)


def compute_lap_time(
    circuit: Circuit,
    lap: int,
    compound_key: str,
    tire_age: int,
    is_pit_lap: bool = False,
    is_safety_car: bool = False,
    track_wetness: float = 0.0,
    in_dirty_air: bool = False,
    driver_pace_offset_sec: float = 0.0,
    noise_sec: float = 0.0,
) -> float:
    """
    Compute total elapsed time (seconds) for a single race lap.
    """
    fuel_pen = compute_fuel_penalty(circuit, lap)
    tire_pen = compute_tire_delta(compound_key, tire_age, circuit, track_wetness)

    raw_pace = circuit.base_lap_time_sec + fuel_pen + tire_pen + driver_pace_offset_sec

    if track_wetness > 0.0:
        raw_pace += 4.5 * track_wetness

    if in_dirty_air and not is_safety_car and not is_pit_lap:
        raw_pace += circuit.dirty_air_penalty_sec

    if is_safety_car:
        raw_pace = circuit.base_lap_time_sec * 1.38

    pit_pen = compute_pit_loss(circuit, is_safety_car=is_safety_car) if is_pit_lap else 0.0

    return raw_pace + pit_pen + noise_sec


@dataclass
class CarStrategyPlan:
    """Defines a starting compound and scheduled pit stops {lap_number: new_compound}."""
    name: str
    start_compound: str
    pit_stops: Dict[int, str] = field(default_factory=dict)
    driver_pace_offset_sec: float = 0.0


def simulate_race_trace(
    circuit: Circuit,
    player_plan: CarStrategyPlan,
    rival_plan: CarStrategyPlan,
    safety_car_laps: Optional[List[int]] = None,
    wetness_by_lap: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None,
    stochastic: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Simulate a head-to-head race between Player and Rival over all `circuit.total_laps`.
    """
    n_laps = circuit.total_laps
    sc_set = set(safety_car_laps or [])
    wetness = wetness_by_lap if wetness_by_lap is not None else [0.0] * n_laps

    p_comp = player_plan.start_compound
    r_comp = rival_plan.start_compound
    p_age = 0
    r_age = 0

    p_lap_times = np.zeros(n_laps)
    r_lap_times = np.zeros(n_laps)
    p_cum_time = np.zeros(n_laps)
    r_cum_time = np.zeros(n_laps)
    gap_to_rival = np.zeros(n_laps)
    p_compounds_hist: List[str] = []
    r_compounds_hist: List[str] = []
    p_tire_ages = np.zeros(n_laps, dtype=int)
    r_tire_ages = np.zeros(n_laps, dtype=int)

    overtake_threshold_sec = 0.35 + 1.45 * circuit.overtaking_difficulty
    prev_gap = 0.0

    for idx in range(n_laps):
        lap_num = idx + 1
        is_sc = lap_num in sc_set
        w_lap = wetness[idx] if idx < len(wetness) else 0.0

        p_pit = lap_num in player_plan.pit_stops
        r_pit = lap_num in rival_plan.pit_stops

        p_dirty = False
        r_dirty = False
        if not is_sc and lap_num > 1:
            p_raw_est = compute_tire_delta(p_comp, p_age, circuit, w_lap) + player_plan.driver_pace_offset_sec
            r_raw_est = compute_tire_delta(r_comp, r_age, circuit, w_lap) + rival_plan.driver_pace_offset_sec

            if -1.0 < prev_gap < 0.0:
                pace_adv = r_raw_est - p_raw_est
                if pace_adv < overtake_threshold_sec:
                    p_dirty = True
            elif 0.0 < prev_gap < 1.0:
                pace_adv = p_raw_est - r_raw_est
                if pace_adv < overtake_threshold_sec:
                    r_dirty = True

        p_noise = float(rng.normal(0.0, 0.14)) if (stochastic and rng is not None and not is_sc) else 0.0
        r_noise = float(rng.normal(0.0, 0.14)) if (stochastic and rng is not None and not is_sc) else 0.0

        p_lt = compute_lap_time(
            circuit=circuit,
            lap=lap_num,
            compound_key=p_comp,
            tire_age=p_age,
            is_pit_lap=p_pit,
            is_safety_car=is_sc,
            track_wetness=w_lap,
            in_dirty_air=p_dirty,
            driver_pace_offset_sec=player_plan.driver_pace_offset_sec,
            noise_sec=p_noise,
        )
        r_lt = compute_lap_time(
            circuit=circuit,
            lap=lap_num,
            compound_key=r_comp,
            tire_age=r_age,
            is_pit_lap=r_pit,
            is_safety_car=is_sc,
            track_wetness=w_lap,
            in_dirty_air=r_dirty,
            driver_pace_offset_sec=rival_plan.driver_pace_offset_sec,
            noise_sec=r_noise,
        )

        p_lap_times[idx] = p_lt
        r_lap_times[idx] = r_lt
        p_compounds_hist.append(p_comp)
        r_compounds_hist.append(r_comp)
        p_tire_ages[idx] = p_age + 1
        r_tire_ages[idx] = r_age + 1

        p_cum_time[idx] = (p_cum_time[idx - 1] if idx > 0 else 0.0) + p_lt
        r_cum_time[idx] = (r_cum_time[idx - 1] if idx > 0 else 0.0) + r_lt

        if is_sc:
            raw_gap = r_cum_time[idx] - p_cum_time[idx]
            compressed_gap = np.clip(raw_gap, -0.6, 0.6)
            if raw_gap >= 0:
                r_cum_time[idx] = p_cum_time[idx] + compressed_gap
            else:
                p_cum_time[idx] = r_cum_time[idx] + abs(compressed_gap)

        prev_gap = r_cum_time[idx] - p_cum_time[idx]
        gap_to_rival[idx] = prev_gap

        if p_pit:
            p_comp = player_plan.pit_stops[lap_num]
            p_age = 0
        else:
            p_age += 1

        if r_pit:
            r_comp = rival_plan.pit_stops[lap_num]
            r_age = 0
        else:
            r_age += 1

    return {
        "laps": np.arange(1, n_laps + 1),
        "player_lap_times": p_lap_times,
        "rival_lap_times": r_lap_times,
        "player_cum_time": p_cum_time,
        "rival_cum_time": r_cum_time,
        "gap_to_rival": gap_to_rival,
        "player_compounds": np.array(p_compounds_hist),
        "rival_compounds": np.array(r_compounds_hist),
        "player_tire_ages": p_tire_ages,
        "rival_tire_ages": r_tire_ages,
        "final_gap_sec": float(gap_to_rival[-1]),
    }


if __name__ == "__main__":
    circuits = load_circuits()
    print(f"Loaded {len(circuits)} calibrated 2026 F1 circuits:")
    for k, c in sorted(circuits.items(), key=lambda x: x[1].round):
        print(
            f"  R{c.round:02d} {c.name:<42} | Laps: {c.total_laps:2d} | "
            f"Base: {c.base_lap_time_sec:6.2f}s | PitLoss: {c.pit_loss_sec:5.2f}s | {c.telemetry_source}"
        )