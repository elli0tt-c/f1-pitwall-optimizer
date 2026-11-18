"""
engine/physics.py
Lap-by-lap motorsport race physics engine.

Models:
- Fuel mass burn and longitudinal/lateral lap-time sensitivity
- Compound-specific tire degradation (linear wear + quadratic thermal/rubber cliff)
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
    deg_per_lap_sec: float      # Linear lap-time degradation rate (s/lap)
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
    """Circuit boundary conditions loaded from data/circuits.json."""
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


def load_circuits(json_path: str = "data/circuits.json") -> Dict[str, Circuit]:
    """Load all track presets into strongly-typed Circuit dataclasses."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        key: Circuit(key=key, **cfg)
        for key, cfg in raw.items()
    }


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

    Includes:
    1. Base compound stiffness offset
    2. Circuit-scaled linear rubber wear
    3. Quadratic thermal/wear cliff beyond `cliff_lap`
    4. Wet/dry compound mismatch penalty (aquaplaning or overheating)
    """
    comp = COMPOUNDS[compound_key]
    effective_age = max(0, tire_age)

    # Linear degradation scaled by track abrasiveness/energy
    linear_deg = comp.deg_per_lap_sec * circuit.tire_wear_multiplier * effective_age

    # Non-linear performance cliff once tire exceeds thermal/wear life
    adjusted_cliff = max(5.0, comp.cliff_lap / max(0.5, circuit.tire_wear_multiplier ** 0.5))
    over_cliff = max(0.0, effective_age - adjusted_cliff)
    cliff_deg = comp.cliff_severity * (over_cliff ** 2)

    # Wetness mismatch penalty
    wetness_penalty = 0.0
    if track_wetness < comp.optimal_wetness_min:
        # Wet/Inter tires overheating and graining on a drying track
        deficit = comp.optimal_wetness_min - track_wetness
        wetness_penalty = 18.0 * (deficit ** 1.5)
    elif track_wetness > comp.optimal_wetness_max:
        # Slicks or Inters losing contact patch / aquaplaning in standing water
        excess = track_wetness - comp.optimal_wetness_max
        wetness_penalty = 32.0 * (excess ** 1.6)

    return comp.base_offset_sec + linear_deg + cliff_deg + wetness_penalty


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

    # Base race pace (dry clear air)
    raw_pace = circuit.base_lap_time_sec + fuel_pen + tire_pen + driver_pace_offset_sec

    # Wet track naturally slows even the optimal wet tire slightly
    if track_wetness > 0.0:
        raw_pace += 4.5 * track_wetness

    # Wake aerodynamics penalty if stuck behind a car on green-flag laps
    if in_dirty_air and not is_safety_car and not is_pit_lap:
        raw_pace += circuit.dirty_air_penalty_sec

    # Neutralized Safety Car lap runs ~38% slower than green-flag base pace
    if is_safety_car:
        raw_pace = circuit.base_lap_time_sec * 1.38

    # Add pit-lane transit + stationary service delta if pitting on this lap
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

    Models interactive traffic: if a car catches its rival within 1.0s, it incurs
    dirty-air loss unless its net pace advantage overcomes `circuit.overtaking_difficulty`.
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
    gap_to_rival = np.zeros(n_laps)  # Positive = Player ahead of Rival (seconds)
    p_compounds_hist: List[str] = []
    r_compounds_hist: List[str] = []
    p_tire_ages = np.zeros(n_laps, dtype=int)
    r_tire_ages = np.zeros(n_laps, dtype=int)

    # Pace delta required to execute an overtake (seconds/lap)
    overtake_threshold_sec = 0.35 + 1.45 * circuit.overtaking_difficulty

    prev_gap = 0.0  # Start side-by-side at t = 0

    for idx in range(n_laps):
        lap_num = idx + 1
        is_sc = lap_num in sc_set
        w_lap = wetness[idx] if idx < len(wetness) else 0.0

        # Check scheduled pit stops at the end of this lap
        p_pit = lap_num in player_plan.pit_stops
        r_pit = lap_num in rival_plan.pit_stops

        # Determine if trailing car is caught in dirty air (< 1.0s behind)
        p_dirty = False
        r_dirty = False
        if not is_sc and lap_num > 1:
            p_raw_est = compute_tire_delta(p_comp, p_age, circuit, w_lap) + player_plan.driver_pace_offset_sec
            r_raw_est = compute_tire_delta(r_comp, r_age, circuit, w_lap) + rival_plan.driver_pace_offset_sec

            if -1.0 < prev_gap < 0.0:
                # Player is trailing Rival within 1.0s
                pace_adv = r_raw_est - p_raw_est
                if pace_adv < overtake_threshold_sec:
                    p_dirty = True
            elif 0.0 < prev_gap < 1.0:
                # Rival is trailing Player within 1.0s
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

        # Record telemetry before tire swap takes effect for next lap
        p_lap_times[idx] = p_lt
        r_lap_times[idx] = r_lt
        p_compounds_hist.append(p_comp)
        r_compounds_hist.append(r_comp)
        p_tire_ages[idx] = p_age + 1
        r_tire_ages[idx] = r_age + 1

        p_cum_time[idx] = (p_cum_time[idx - 1] if idx > 0 else 0.0) + p_lt
        r_cum_time[idx] = (r_cum_time[idx - 1] if idx > 0 else 0.0) + r_lt

        # Under Safety Car, field compresses behind leader (gaps shrink toward ~0.6s)
        if is_sc:
            raw_gap = r_cum_time[idx] - p_cum_time[idx]
            compressed_gap = np.clip(raw_gap, -0.6, 0.6)
            if raw_gap >= 0:
                r_cum_time[idx] = p_cum_time[idx] + compressed_gap
            else:
                p_cum_time[idx] = r_cum_time[idx] + abs(compressed_gap)

        prev_gap = r_cum_time[idx] - p_cum_time[idx]
        gap_to_rival[idx] = prev_gap

        # Update tire state for next lap
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
    # Quick physics test bench: Compare 1-stop vs 2-stop at Silverstone
    circuits = load_circuits()
    silverstone = circuits["silverstone"]

    one_stop = CarStrategyPlan(name="1-Stop (M->H)", start_compound="MEDIUM", pit_stops={22: "HARD"})
    two_stop = CarStrategyPlan(name="2-Stop (S->M->M)", start_compound="SOFT", pit_stops={15: "MEDIUM", 34: "MEDIUM"})

    res = simulate_race_trace(silverstone, player_plan=two_stop, rival_plan=one_stop)
    winner = "2-Stop" if res["final_gap_sec"] > 0 else "1-Stop"
    print(f"Silverstone 52-Lap Test: {winner} wins by {abs(res['final_gap_sec']):.2f}s")
    print(f"2-Stop Total Time: {res['player_cum_time'][-1]:.2f}s | 1-Stop Total Time: {res['rival_cum_time'][-1]:.2f}s")