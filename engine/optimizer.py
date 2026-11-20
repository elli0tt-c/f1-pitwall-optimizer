"""
engine/optimizer.py
Motorsport Strategy Optimization & Stochastic Risk Engine.

Combines:
1. Exact Backward Dynamic Programming (Bellman optimality) to find the minimum-time
   dry-weather pit window and compound sequence (enforcing the 2-compound FIA rule).
2. Mid-Race Re-Solver: dynamically recomputes optimal remaining stints from any live
   race state (e.g., reacting instantaneously to a Safety Car or Rival Undercut).
3. Custom & Actual Race Strategy Builder: constructs and validates user-defined or
   historical Grand Prix driver strategies against FIA sporting regulations.
4. Monte Carlo Rollout Engine: stress-tests candidate strategies across hundreds of
   stochastic race futures (random Safety Cars, rain showers, and lap-time variance).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from engine.physics import (
    COMPOUNDS,
    DRY_COMPOUNDS,
    CarStrategyPlan,
    Circuit,
    compute_lap_time,
    simulate_race_trace,
)

# Preference order for breaking mathematical commutativity ties (softer start = better launch grip)
_COMPOUND_START_PRIORITY: Dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTER": 3,
    "WET": 4,
}


@dataclass
class DPOptimalResult:
    """Output of the Dynamic Programming race-strategy solver."""
    plan: CarStrategyPlan
    expected_total_time_sec: float
    pit_laps: List[int]
    stint_summary: str
    top_alternatives: List[Tuple[str, float, CarStrategyPlan]]


@dataclass
class CustomStrategyResult:
    """Validated user-constructed or real-world pit-wall strategy plan."""
    plan: CarStrategyPlan
    stint_summary: str
    is_fia_legal_dry: bool
    warnings: List[str]


@dataclass
class ActualDriverStrategy:
    """Real Grand Prix driver classification and stint sequence extracted from FastF1."""
    position: int
    grid_position: int
    driver_code: str
    driver_name: str
    team: str
    status: str
    laps_completed: int
    num_stops: int
    start_compound: str
    stops: List[Tuple[int, str]]
    stint_sequence: List[str]
    strategy_summary: str
    is_dry_only: bool

    @property
    def display_label(self) -> str:
        pos_tag = f"P{self.position}" if self.position < 90 else "DNF"
        return f"{pos_tag} {self.driver_code} ({self.team}) — {self.strategy_summary}"


def _canonicalize_compound_key(raw: str) -> str:
    """Map FastF1 compound labels ('INTERMEDIATE') to physics engine keys ('INTER')."""
    c = str(raw or "").strip().upper()
    if c in ("INTERMEDIATE", "INTER"):
        return "INTER"
    if c in COMPOUNDS:
        return c
    return "MEDIUM"


def load_actual_strategies(
    filepath: str = "data/actual_strategies.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Load extracted Grand Prix driver strategies from JSON.
    Returns a dict keyed by circuit_key containing 'source', 'total_laps', and
    'drivers' (List[ActualDriverStrategy]).
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception:
        return {}

    parsed: Dict[str, Dict[str, Any]] = {}
    for circuit_key, entry in raw_data.items():
        driver_objs: List[ActualDriverStrategy] = []
        for d in entry.get("drivers", []):
            start_c = _canonicalize_compound_key(d.get("start_compound", "MEDIUM"))
            pit_laps = d.get("pit_laps", [])
            next_comps = [_canonicalize_compound_key(c) for c in d.get("compounds", [])]
            stops = list(zip([int(l) for l in pit_laps], next_comps))
            stint_seq = [_canonicalize_compound_key(c) for c in d.get("stint_sequence", [start_c])]

            driver_objs.append(
                ActualDriverStrategy(
                    position=int(d.get("position", 99)),
                    grid_position=int(d.get("grid_position", 0)),
                    driver_code=str(d.get("driver_code", "DRV")),
                    driver_name=str(d.get("driver_name", "Driver")),
                    team=str(d.get("team", "Team")),
                    status=str(d.get("status", "Finished")),
                    laps_completed=int(d.get("laps_completed", entry.get("total_laps", 50))),
                    num_stops=len(stops),
                    start_compound=start_c,
                    stops=stops,
                    stint_sequence=stint_seq,
                    strategy_summary=str(d.get("strategy_summary", "")),
                    is_dry_only=bool(d.get("is_dry_only", True)),
                )
            )

        parsed[circuit_key] = {
            "source": str(entry.get("source", "FastF1")),
            "total_laps": int(entry.get("total_laps", 50)),
            "drivers": driver_objs,
        }

    return parsed


def build_custom_strategy_plan(
    circuit: Circuit,
    start_compound: str,
    stops: List[Tuple[int, str]],
    driver_pace_offset_sec: float = 0.0,
    plan_prefix: str = "Custom",
) -> CustomStrategyResult:
    """
    Construct and validate a custom user-defined or historical pit-wall strategy.

    Args:
        circuit: Active Grand Prix circuit configuration.
        start_compound: Starting tire compound key (e.g., "SOFT", "MEDIUM", "HARD").
        stops: List of (pit_lap, next_compound) tuples.
        driver_pace_offset_sec: Driver/car pace delta vs. rival (s/lap).
        plan_prefix: Label prefix for the CarStrategyPlan (e.g., "Custom" or "P1 NOR").

    Returns:
        CustomStrategyResult containing the executable CarStrategyPlan, formatted
        stint label, FIA 2-compound compliance boolean, and any validation warnings.
    """
    warnings: List[str] = []
    n_laps = circuit.total_laps

    start_compound = _canonicalize_compound_key(start_compound)
    if start_compound not in COMPOUNDS:
        start_compound = "MEDIUM"
        warnings.append("Invalid starting compound defaulted to MEDIUM.")

    # Sort stops chronologically and clamp pit laps to valid race window [1, n_laps - 1]
    cleaned_stops: Dict[int, str] = {}
    seen_laps = set()
    for raw_lap, comp in sorted(stops, key=lambda item: item[0]):
        clamped_lap = max(1, min(n_laps - 1, int(raw_lap)))
        if clamped_lap != raw_lap:
            warnings.append(f"Pit lap {raw_lap} adjusted to valid race window (Lap {clamped_lap}).")
        if clamped_lap in seen_laps:
            warnings.append(f"Duplicate pit stop on Lap {clamped_lap} merged (last compound kept).")
        seen_laps.add(clamped_lap)

        valid_comp = _canonicalize_compound_key(comp)
        cleaned_stops[clamped_lap] = valid_comp

    # Check FIA Sporting Regulation: at least 2 distinct dry compounds required in a dry race
    compounds_used = [start_compound] + [cleaned_stops[l] for l in sorted(cleaned_stops)]
    distinct_dry_used = {c for c in compounds_used if c in DRY_COMPOUNDS}
    has_wet_compound = any(c in ("INTER", "WET") for c in compounds_used)
    is_fia_legal_dry = (len(distinct_dry_used) >= 2) or has_wet_compound

    if not is_fia_legal_dry:
        warnings.append(
            "FIA Regulation Warning: Fewer than 2 distinct dry compounds scheduled "
            "(subject to post-race disqualification in dry conditions)."
        )

    stint_codes = [COMPOUNDS[c].code for c in compounds_used]
    stint_label = f"{len(cleaned_stops)}-Stop ({'→'.join(stint_codes)})"
    if cleaned_stops:
        laps_str = ", ".join(f"L{l}" for l in sorted(cleaned_stops))
        stint_label += f" [{laps_str}]"

    plan = CarStrategyPlan(
        name=f"{plan_prefix}: {stint_label}",
        start_compound=start_compound,
        pit_stops=cleaned_stops,
        driver_pace_offset_sec=driver_pace_offset_sec,
    )

    return CustomStrategyResult(
        plan=plan,
        stint_summary=stint_label,
        is_fia_legal_dry=is_fia_legal_dry,
        warnings=warnings,
    )


def build_actual_driver_plan(
    circuit: Circuit,
    driver_strategy: ActualDriverStrategy,
    driver_pace_offset_sec: float = 0.0,
) -> CustomStrategyResult:
    """
    Convert a real Grand Prix driver's FastF1 stint history into a validated
    CarStrategyPlan for head-to-head simulation and DP benchmarking.
    """
    pos_tag = f"P{driver_strategy.position}" if driver_strategy.position < 90 else "DNF"
    prefix = f"{pos_tag} {driver_strategy.driver_code}"
    return build_custom_strategy_plan(
        circuit=circuit,
        start_compound=driver_strategy.start_compound,
        stops=driver_strategy.stops,
        driver_pace_offset_sec=driver_pace_offset_sec,
        plan_prefix=prefix,
    )


def _extract_trajectory_for_exact_stops(
    comp_list: List[str],
    lap_cost: np.ndarray,
    start_lap: int,
    n_laps: int,
    remaining_laps: int,
    max_age: int,
    target_stops: int,
    is_safety_car_now: bool,
    pit_loss_sec: float,
    start_choices: List[Tuple[int, int, int]],
    driver_pace_offset_sec: float,
) -> List[Tuple[str, float, CarStrategyPlan]]:
    """
    Solve Backward DP constrained to finish with `target_stops` total pit stops
    and extract the optimal trajectory for each valid starting compound.
    """
    n_comp = len(comp_list)
    INF = 1e12

    V = np.full((remaining_laps + 1, n_comp, max_age, 2, target_stops + 1), INF, dtype=np.float64)
    policy = np.full((remaining_laps, n_comp, max_age, 2, target_stops + 1), -2, dtype=np.int8)

    # Terminal condition: checkered flag reached with >= 2 dry compounds AND exact target_stops
    V[remaining_laps, :, :, 1, target_stops] = 0.0

    for step in range(remaining_laps - 1, -1, -1):
        lap_num = start_lap + step
        l_idx = lap_num - 1
        sc_flag = is_safety_car_now and (lap_num == start_lap)
        pit_delta = pit_loss_sec * (0.55 if sc_flag else 1.0)

        for c_idx in range(n_comp):
            for age in range(max_age - 1):
                base_t = lap_cost[l_idx, c_idx, age]
                if base_t >= INF:
                    continue

                for used_two in (0, 1):
                    for stops in range(target_stops + 1):
                        # Option A: Stay out on current tires
                        best_val = base_t + V[step + 1, c_idx, age + 1, used_two, stops]
                        best_act = -1

                        # Option B: Pit at end of this lap (not allowed on final lap)
                        if lap_num < n_laps and stops < target_stops:
                            for c_next in range(n_comp):
                                next_used_two = 1 if (used_two == 1 or c_next != c_idx) else 0
                                cand_val = base_t + pit_delta + V[step + 1, c_next, 0, next_used_two, stops + 1]
                                if cand_val < best_val:
                                    best_val = cand_val
                                    best_act = c_next

                        V[step, c_idx, age, used_two, stops] = best_val
                        policy[step, c_idx, age, used_two, stops] = best_act

    extracted: List[Tuple[str, float, CarStrategyPlan]] = []
    for c_start_idx, init_age, init_used in start_choices:
        total_t = V[0, c_start_idx, init_age, init_used, 0]
        if total_t >= INF:
            continue

        c_curr = c_start_idx
        age_curr = init_age
        used_curr = init_used
        stops_curr = 0
        pit_stops: Dict[int, str] = {}

        for step in range(remaining_laps):
            lap_num = start_lap + step
            act = int(policy[step, c_curr, age_curr, used_curr, stops_curr])
            if act == -1:
                age_curr += 1
            elif act >= 0:
                new_comp_name = comp_list[act]
                pit_stops[lap_num] = new_comp_name
                used_curr = 1 if (used_curr == 1 or act != c_curr) else 0
                c_curr = act
                age_curr = 0
                stops_curr += 1

        start_name = comp_list[c_start_idx]
        stint_codes = [COMPOUNDS[start_name].code] + [COMPOUNDS[pit_stops[l]].code for l in sorted(pit_stops)]
        stint_label = f"{len(pit_stops)}-Stop ({'→'.join(stint_codes)})"
        if pit_stops:
            laps_str = ", ".join(f"L{l}" for l in sorted(pit_stops))
            stint_label += f" [{laps_str}]"

        plan = CarStrategyPlan(
            name=f"DP Optimal: {stint_label}",
            start_compound=start_name,
            pit_stops=pit_stops,
            driver_pace_offset_sec=driver_pace_offset_sec,
        )
        extracted.append((stint_label, float(total_t), plan))

    return extracted


def solve_optimal_strategy_dp(
    circuit: Circuit,
    start_lap: int = 1,
    current_compound: Optional[str] = None,
    current_tire_age: int = 0,
    used_different_compound: bool = False,
    is_safety_car_now: bool = False,
    max_stops: int = 3,
    driver_pace_offset_sec: float = 0.0,
) -> DPOptimalResult:
    """
    Solve for the globally optimal pit-stop lap(s) and tire compound sequence from
    `start_lap` to `circuit.total_laps` using exact Backward Dynamic Programming.

    Enforces the sporting regulation requiring at least two distinct dry compounds
    during a dry race, and deduplicates mathematically commutative stint permutations
    (e.g., M→H vs. H→M) in favor of the softer-start race-launch sequence.
    """
    n_laps = circuit.total_laps
    remaining_laps = n_laps - start_lap + 1
    comp_list = list(DRY_COMPOUNDS)
    n_comp = len(comp_list)
    max_age = n_laps + 1

    # Precompute single-lap costs: lap_cost[lap_idx, comp_idx, age]
    lap_cost = np.full((n_laps, n_comp, max_age), np.inf, dtype=np.float64)
    for l_idx in range(start_lap - 1, n_laps):
        lap_num = l_idx + 1
        sc_flag = is_safety_car_now and (lap_num == start_lap)
        for c_idx, c_name in enumerate(comp_list):
            max_feasible_age = l_idx + current_tire_age + 1
            for age in range(min(max_age, max_feasible_age + 1)):
                lap_cost[l_idx, c_idx, age] = compute_lap_time(
                    circuit=circuit,
                    lap=lap_num,
                    compound_key=c_name,
                    tire_age=age,
                    is_pit_lap=False,
                    is_safety_car=sc_flag,
                    track_wetness=0.0,
                    in_dirty_air=False,
                    driver_pace_offset_sec=driver_pace_offset_sec,
                )

    start_choices = (
        [(comp_list.index(current_compound), current_tire_age, 1 if used_different_compound else 0)]
        if current_compound is not None
        else [(c_i, 0, 0) for c_i in range(n_comp)]
    )

    min_stops = 0 if (current_compound is not None and used_different_compound) else 1
    raw_candidates: List[Tuple[str, float, CarStrategyPlan]] = []

    for stops_target in range(min_stops, max_stops + 1):
        raw_candidates.extend(
            _extract_trajectory_for_exact_stops(
                comp_list=comp_list,
                lap_cost=lap_cost,
                start_lap=start_lap,
                n_laps=n_laps,
                remaining_laps=remaining_laps,
                max_age=max_age,
                target_stops=stops_target,
                is_safety_car_now=is_safety_car_now,
                pit_loss_sec=circuit.pit_loss_sec,
                start_choices=start_choices,
                driver_pace_offset_sec=driver_pace_offset_sec,
            )
        )

    # Sort primarily by total race time, secondarily by softer starting compound
    raw_candidates.sort(
        key=lambda x: (round(x[1], 6), _COMPOUND_START_PRIORITY.get(x[2].start_compound, 99))
    )

    # Deduplicate commutative mirror strategies (e.g. M→H and H→M with identical race time)
    deduped_candidates: List[Tuple[str, float, CarStrategyPlan]] = []
    seen_canonical_sets: Dict[Tuple[int, Tuple[str, ...]], float] = {}

    for label, total_t, plan in raw_candidates:
        used_seq = [plan.start_compound] + [plan.pit_stops[l] for l in sorted(plan.pit_stops)]
        canonical_key = (len(plan.pit_stops), tuple(sorted(used_seq)))

        if canonical_key in seen_canonical_sets:
            # Skip commutative mirror duplicates that have the exact same total time
            if abs(total_t - seen_canonical_sets[canonical_key]) < 1e-4:
                continue
        else:
            seen_canonical_sets[canonical_key] = total_t

        deduped_candidates.append((label, total_t, plan))

    best_label, best_time, best_plan = deduped_candidates[0]

    return DPOptimalResult(
        plan=best_plan,
        expected_total_time_sec=best_time,
        pit_laps=sorted(best_plan.pit_stops.keys()),
        stint_summary=best_label,
        top_alternatives=deduped_candidates,
    )


def generate_standard_rival_strategies(circuit: Circuit) -> List[CarStrategyPlan]:
    """
    Generate realistic competing pit-wall strategies (Conservative 1-Stop,
    Aggressive 2-Stop, and Early Undercut) tailored to the circuit's lap count.
    """
    n = circuit.total_laps
    p1_one = max(12, int(round(n * 0.42)))
    p1_two = max(10, int(round(n * 0.28)))
    p2_two = max(p1_two + 10, int(round(n * 0.64)))
    p_under = max(8, p1_one - 4)

    return [
        CarStrategyPlan(
            name=f"1-Stop Standard (M→H) [L{p1_one}]",
            start_compound="MEDIUM",
            pit_stops={p1_one: "HARD"},
        ),
        CarStrategyPlan(
            name=f"2-Stop Aggressive (S→M→M) [L{p1_two}, L{p2_two}]",
            start_compound="SOFT",
            pit_stops={p1_two: "MEDIUM", p2_two: "MEDIUM"},
        ),
        CarStrategyPlan(
            name=f"1-Stop Undercut (S→H) [L{p_under}]",
            start_compound="SOFT",
            pit_stops={p_under: "HARD"},
        ),
    ]


@dataclass
class MonteCarloSummary:
    """Aggregated statistical outcomes across N stochastic race rollouts."""
    strategy_name: str
    win_probability_pct: float
    mean_final_gap_sec: float
    std_final_gap_sec: float
    p10_gap_sec: float
    p50_gap_sec: float
    p90_gap_sec: float
    mean_race_time_sec: float
    all_gaps_sec: np.ndarray
    all_gap_traces: np.ndarray  # Shape: (n_sims, total_laps)


def _sample_race_environment(
    circuit: Circuit,
    rng: np.random.Generator,
    enable_safety_cars: bool = True,
    enable_weather: bool = True,
) -> Tuple[List[int], List[float]]:
    """
    Sample a stochastic race environment: random Safety Car deployment windows
    and transient rain showers.
    """
    n = circuit.total_laps
    sc_laps: List[int] = []
    wetness = [0.0] * n

    if enable_safety_cars:
        lap = 3  # No SC on laps 1-2
        while lap <= n - 3:
            if rng.random() < circuit.safety_car_prob_per_lap:
                duration = int(rng.integers(3, 6))  # SC stays out 3 to 5 laps
                for d in range(duration):
                    if lap + d <= n:
                        sc_laps.append(lap + d)
                lap += duration + 6  # Cooldown window
            else:
                lap += 1

    if enable_weather and rng.random() < circuit.rain_prob:
        # Simulate a passing rain cell of 8-18 laps
        rain_start = int(rng.integers(max(4, n // 5), max(6, int(n * 0.75))))
        rain_len = int(rng.integers(8, 19))
        peak_wetness = float(rng.uniform(0.28, 0.78))
        for idx in range(n):
            lap_num = idx + 1
            if rain_start <= lap_num < rain_start + rain_len:
                progress = (lap_num - rain_start) / max(1, rain_len - 1)
                # Smooth bell-shaped intensity curve
                wetness[idx] = peak_wetness * float(np.sin(np.pi * progress))

    return sc_laps, wetness


def _adapt_plan_for_rain(plan: CarStrategyPlan, wetness_by_lap: List[float]) -> CarStrategyPlan:
    """
    Reactive pit-wall weather rule: if track wetness crosses the dry/inter crossover
    (0.18), box for Intermediates (or Full Wets if > 0.62); return to Mediums once dry.
    """
    adapted_stops = dict(plan.pit_stops)
    current_comp = plan.start_compound
    n_laps = len(wetness_by_lap)

    for idx in range(n_laps - 1):
        lap_num = idx + 1
        w = wetness_by_lap[idx]

        # Apply any scheduled dry stop first
        if lap_num in adapted_stops:
            current_comp = adapted_stops[lap_num]

        if w > 0.62 and current_comp != "WET":
            adapted_stops[lap_num] = "WET"
            current_comp = "WET"
        elif 0.18 < w <= 0.62 and current_comp not in ("INTER", "WET"):
            adapted_stops[lap_num] = "INTER"
            current_comp = "INTER"
        elif w <= 0.14 and current_comp in ("INTER", "WET"):
            adapted_stops[lap_num] = "MEDIUM"
            current_comp = "MEDIUM"

    return CarStrategyPlan(
        name=plan.name,
        start_compound=plan.start_compound,
        pit_stops=adapted_stops,
        driver_pace_offset_sec=plan.driver_pace_offset_sec,
    )


def run_monte_carlo_evaluation(
    circuit: Circuit,
    player_plan: CarStrategyPlan,
    rival_plan: CarStrategyPlan,
    n_sims: int = 300,
    seed: int = 42,
    enable_safety_cars: bool = True,
    enable_weather: bool = True,
) -> MonteCarloSummary:
    """
    Run `n_sims` Monte Carlo race simulations comparing `player_plan` vs `rival_plan`
    under stochastic Safety Cars, weather cells, and lap-time perturbations.
    """
    rng = np.random.default_rng(seed)
    n_laps = circuit.total_laps

    final_gaps = np.zeros(n_sims, dtype=np.float64)
    total_times = np.zeros(n_sims, dtype=np.float64)
    gap_traces = np.zeros((n_sims, n_laps), dtype=np.float64)

    for i in range(n_sims):
        sc_laps, wetness = _sample_race_environment(
            circuit=circuit,
            rng=rng,
            enable_safety_cars=enable_safety_cars,
            enable_weather=enable_weather,
        )

        # Reactive wet-weather pit calls if rain hits
        p_exec = _adapt_plan_for_rain(player_plan, wetness) if any(w > 0.15 for w in wetness) else player_plan
        r_exec = _adapt_plan_for_rain(rival_plan, wetness) if any(w > 0.15 for w in wetness) else rival_plan

        trace = simulate_race_trace(
            circuit=circuit,
            player_plan=p_exec,
            rival_plan=r_exec,
            safety_car_laps=sc_laps,
            wetness_by_lap=wetness,
            rng=rng,
            stochastic=True,
        )

        final_gaps[i] = trace["final_gap_sec"]
        total_times[i] = float(trace["player_cum_time"][-1])
        gap_traces[i] = trace["gap_to_rival"]

    win_prob = float(np.mean(final_gaps > 0.0) * 100.0)

    return MonteCarloSummary(
        strategy_name=player_plan.name,
        win_probability_pct=win_prob,
        mean_final_gap_sec=float(np.mean(final_gaps)),
        std_final_gap_sec=float(np.std(final_gaps)),
        p10_gap_sec=float(np.percentile(final_gaps, 10)),
        p50_gap_sec=float(np.percentile(final_gaps, 50)),
        p90_gap_sec=float(np.percentile(final_gaps, 90)),
        mean_race_time_sec=float(np.mean(total_times)),
        all_gaps_sec=final_gaps,
        all_gap_traces=gap_traces,
    )


if __name__ == "__main__":
    from engine.physics import load_circuits

    circuits = load_circuits()
    actuals = load_actual_strategies()
    for key in ("monza", "silverstone", "monaco"):
        c = circuits[key]
        dp_res = solve_optimal_strategy_dp(c)
        rivals = generate_standard_rival_strategies(c)
        mc = run_monte_carlo_evaluation(c, dp_res.plan, rivals[0], n_sims=250)
        p1_info = ""
        if key in actuals and actuals[key]["drivers"]:
            p1 = actuals[key]["drivers"][0]
            p1_info = f" | Real P1: {p1.driver_code} {p1.strategy_summary}"
        print(
            f"[{c.name}] DP Optimal: {dp_res.stint_summary} "
            f"| Est. Time: {dp_res.expected_total_time_sec:.2f}s "
            f"| Win Prob vs {rivals[0].name}: {mc.win_probability_pct:.1f}%{p1_info}"
        )