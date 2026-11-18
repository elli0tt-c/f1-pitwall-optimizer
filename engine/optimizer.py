"""
engine/optimizer.py
Motorsport Strategy Optimization & Stochastic Risk Engine.

Combines:
1. Exact Backward Dynamic Programming (Bellman optimality) to find the minimum-time
   dry-weather pit window and compound sequence (enforcing the 2-compound FIA rule).
2. Mid-Race Re-Solver: dynamically recomputes optimal remaining stints from any live
   race state (e.g., reacting instantaneously to a Safety Car or Rival Undercut).
3. Monte Carlo Rollout Engine: stress-tests candidate strategies across hundreds of
   stochastic race futures (random Safety Cars, rain showers, and lap-time variance).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from engine.physics import (
    COMPOUNDS,
    DRY_COMPOUNDS,
    CarStrategyPlan,
    Circuit,
    compute_lap_time,
    simulate_race_trace,
)


@dataclass
class DPOptimalResult:
    """Output of the Dynamic Programming race-strategy solver."""
    plan: CarStrategyPlan
    expected_total_time_sec: float
    pit_laps: List[int]
    stint_summary: str
    top_alternatives: List[Tuple[str, float, CarStrategyPlan]]


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
    during a dry race.
    """
    n_laps = circuit.total_laps
    remaining_laps = n_laps - start_lap + 1
    comp_list = list(DRY_COMPOUNDS)
    n_comp = len(comp_list)
    max_age = n_laps + 1

    # Precompute single-lap costs: lap_cost[lap_idx, comp_idx, age]
    # Note: lap_idx is 0-indexed (0 -> Lap 1)
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

    # DP value table: V[step, comp_idx, age, used_two_comps, stops_used]
    # step goes from 0 (start_lap) to remaining_laps (finish)
    INF = 1e12
    V = np.full((remaining_laps + 1, n_comp, max_age, 2, max_stops + 1), INF, dtype=np.float64)

    # Policy backpointers: action = -1 (stay out) or c_next in {0, 1, 2} (pit to c_next)
    policy = np.full((remaining_laps, n_comp, max_age, 2, max_stops + 1), -2, dtype=np.int8)

    # Terminal boundary condition at checkered flag: must have used >= 2 dry compounds
    V[remaining_laps, :, :, 1, :] = 0.0

    # Backward induction from final lap back to start_lap
    for step in range(remaining_laps - 1, -1, -1):
        lap_num = start_lap + step
        l_idx = lap_num - 1
        sc_flag = is_safety_car_now and (lap_num == start_lap)
        pit_delta = circuit.pit_loss_sec * (0.55 if sc_flag else 1.0)

        for c_idx in range(n_comp):
            for age in range(max_age - 1):
                base_t = lap_cost[l_idx, c_idx, age]
                if base_t >= INF:
                    continue

                for used_two in (0, 1):
                    for stops in range(max_stops + 1):
                        # Option A: Stay out on current tires
                        best_val = base_t + V[step + 1, c_idx, age + 1, used_two, stops]
                        best_act = -1

                        # Option B: Pit at end of this lap (not allowed on final lap)
                        if lap_num < n_laps and stops < max_stops:
                            for c_next in range(n_comp):
                                next_used_two = 1 if (used_two == 1 or c_next != c_idx) else 0
                                cand_val = base_t + pit_delta + V[step + 1, c_next, 0, next_used_two, stops + 1]
                                if cand_val < best_val:
                                    best_val = cand_val
                                    best_act = c_next

                        V[step, c_idx, age, used_two, stops] = best_val
                        policy[step, c_idx, age, used_two, stops] = best_act

    # Evaluate starting compound choices if solving from Lap 1
    candidates: List[Tuple[str, float, CarStrategyPlan]] = []

    start_choices = (
        [(comp_list.index(current_compound), current_tire_age, 1 if used_different_compound else 0)]
        if current_compound is not None
        else [(c_i, 0, 0) for c_i in range(n_comp)]
    )

    for c_start_idx, init_age, init_used in start_choices:
        total_t = V[0, c_start_idx, init_age, init_used, 0]
        if total_t >= INF:
            continue

        # Reconstruct optimal trajectory forward
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
        candidates.append((stint_label, float(total_t), plan))

    candidates.sort(key=lambda x: x[1])
    best_label, best_time, best_plan = candidates[0]

    return DPOptimalResult(
        plan=best_plan,
        expected_total_time_sec=best_time,
        pit_laps=sorted(best_plan.pit_stops.keys()),
        stint_summary=best_label,
        top_alternatives=candidates,
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
    for key in ("monza", "silverstone", "monaco"):
        c = circuits[key]
        dp_res = solve_optimal_strategy_dp(c)
        rivals = generate_standard_rival_strategies(c)
        mc = run_monte_carlo_evaluation(c, dp_res.plan, rivals[0], n_sims=250)
        print(
            f"[{c.name}] DP Optimal: {dp_res.stint_summary} "
            f"| Est. Time: {dp_res.expected_total_time_sec:.2f}s "
            f"| Win Prob vs {rivals[0].name}: {mc.win_probability_pct:.1f}% "
            f"(Mean Gap: {mc.mean_final_gap_sec:+.2f}s)"
        )