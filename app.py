"""
app.py
F1 / Endurance Pit-Wall Strategy Simulator (v2.3 — 2026 Season FastF1 & Real Race Results Edition)

Interactive Streamlit + Plotly pit-wall command center featuring:
- All 23 circuits of the 2026 FIA Formula 1 World Championship calibrated via FastF1
- Exact Backward Dynamic Programming (Bellman optimality) solver enforcing the 2-compound rule
- Interactive Custom Pit-Wall Strategy Builder with FIA 2-compound rule validation & DP delta benchmarking
- Real Grand Prix Result Strategy loader (P1–P20 actual FastF1 driver pit strategies vs. DP optimum)
- Live mid-race chaos controls (Safety Car deployment, Rival Undercut, Passing Rain Cell)
- Monte Carlo stochastic race rollout engine with P10–P90 confidence fan chart
- Circuit-specific empirical tire compound degradation & thermal cliff curves
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.optimizer import (
    build_actual_driver_plan,
    build_custom_strategy_plan,
    generate_standard_rival_strategies,
    load_actual_strategies,
    run_monte_carlo_evaluation,
    solve_optimal_strategy_dp,
)
from engine.physics import (
    COMPOUNDS,
    DRY_COMPOUNDS,
    CarStrategyPlan,
    compute_tire_delta,
    load_circuits,
    simulate_race_trace,
)

st.set_page_config(
    page_title="F1 Pit-Wall Strategy Optimizer | 2026 FastF1 Edition",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark pit-wall styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 14px 18px;
    }
    .telemetry-badge {
        display: inline-block;
        background-color: #161B22;
        border: 1px solid #2EA043;
        color: #3FB950;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data
def get_all_circuits(_mtime: float):
    return load_circuits()


@st.cache_data
def get_all_actual_strategies(_mtime: float):
    return load_actual_strategies()


circuits = get_all_circuits(_file_mtime("data/circuits.json"))
actual_strategies_by_circuit = get_all_actual_strategies(_file_mtime("data/actual_strategies.json"))
sorted_keys = sorted(circuits.keys(), key=lambda k: (circuits[k].round, circuits[k].name))

# --- SIDEBAR: PIT-WALL CONTROLS ---
st.sidebar.title("🏎️ Pit-Wall Telemetry & Controls")

selected_key = st.sidebar.selectbox(
    "2026 Grand Prix Circuit (All 23 Rounds)",
    options=sorted_keys,
    index=0,
    format_func=lambda k: f"R{circuits[k].round:02d} — {circuits[k].name} ({circuits[k].country})",
)
circuit = circuits[selected_key]
circuit_actuals = actual_strategies_by_circuit.get(selected_key, {})
actual_drivers = circuit_actuals.get("drivers", [])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Solver & Monte Carlo Settings")
max_stops = st.sidebar.slider("DP Solver Max Allowed Pit Stops", min_value=1, max_value=3, value=2)
driver_pace_offset = st.sidebar.slider(
    "Player Pace Delta vs. Rival (s/lap)",
    min_value=-0.40,
    max_value=0.40,
    value=0.00,
    step=0.05,
    help="Negative = faster car/driver pace than rival.",
)
n_sims = st.sidebar.slider("Monte Carlo Rollouts", min_value=100, max_value=600, value=250, step=50)
enable_sc_mc = st.sidebar.checkbox("Stochastic Safety Cars in Monte Carlo", value=True)
enable_rain_mc = st.sidebar.checkbox("Stochastic Rain Cells in Monte Carlo", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Live Race Chaos Simulator")
chaos_lap = st.sidebar.slider(
    "Current Race Lap (Decision Point)",
    min_value=1,
    max_value=max(2, circuit.total_laps - 5),
    value=min(18, max(2, circuit.total_laps // 3)),
)
inject_sc = st.sidebar.toggle("🟡 Deploy Safety Car on Current Lap", value=False)
inject_undercut = st.sidebar.toggle("⚡ Rival Boxes Early (Undercut Threat)", value=False)
inject_rain = st.sidebar.toggle("🌧️ Passing Rain Shower (+8 Laps)", value=False)

# --- HEADER & TELEMETRY BADGE ---
st.title(f"🏁 R{circuit.round:02d}: {circuit.name}")
st.markdown(
    f'<div class="telemetry-badge">📡 Telemetry Source: {circuit.telemetry_source} '
    f'&nbsp;|&nbsp; {circuit.total_laps} Laps &nbsp;|&nbsp; Base Pace: {circuit.base_lap_time_sec:.2f}s '
    f'&nbsp;|&nbsp; Pit Delta: {circuit.pit_loss_sec:.2f}s</div>',
    unsafe_allow_html=True,
)

# --- SOLVE PRE-RACE DP & RIVAL BENCHMARKS ---
dp_result = solve_optimal_strategy_dp(
    circuit=circuit,
    max_stops=max_stops,
    driver_pace_offset_sec=driver_pace_offset,
)
rival_strategies = generate_standard_rival_strategies(circuit)

mode_options = [
    "🤖 Bellman DP Optimal (Auto)",
    "🛠️ Custom Pit-Wall Strategy (Manual)",
    "🏁 Real Grand Prix Result Strategy",
]

ctrl_col1, ctrl_col2 = st.columns([3, 2])
with ctrl_col1:
    strategy_mode = st.radio(
        "Player Pit-Wall Strategy Mode",
        options=mode_options,
        horizontal=True,
    )
with ctrl_col2:
    rival_idx = st.selectbox(
        "Primary Rival Strategy Benchmark",
        options=list(range(len(rival_strategies))),
        format_func=lambda i: rival_strategies[i].name,
    )
rival_plan = rival_strategies[rival_idx]

# --- CUSTOM OR REAL GRAND PRIX STRATEGY PANEL ---
custom_result = None
selected_actual_driver = None
base_player_plan = dp_result.plan

if strategy_mode == "🛠️ Custom Pit-Wall Strategy (Manual)":
    with st.expander("🛠️ Custom Pit-Wall Strategy Designer", expanded=True):
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            custom_start_comp = st.selectbox(
                "Starting Tire Compound",
                options=list(DRY_COMPOUNDS),
                index=0,
                format_func=lambda c: f"{COMPOUNDS[c].name} ({COMPOUNDS[c].code})",
                key=f"custom_start_{selected_key}",
            )
        with b_col2:
            custom_num_stops = st.number_input(
                "Scheduled Pit Stops",
                min_value=0,
                max_value=3,
                value=1,
                step=1,
                key=f"custom_stops_{selected_key}",
            )

        custom_stops_input = []
        if custom_num_stops > 0:
            stop_cols = st.columns(int(custom_num_stops))
            for s_idx in range(int(custom_num_stops)):
                default_lap = max(
                    2,
                    min(
                        circuit.total_laps - 1,
                        int(round(circuit.total_laps * (s_idx + 1) / (custom_num_stops + 1))),
                    ),
                )
                default_comp_idx = 2 if (s_idx == custom_num_stops - 1 and custom_start_comp != "HARD") else 1
                with stop_cols[s_idx]:
                    st.markdown(f"**Pit Stop #{s_idx + 1}**")
                    p_lap = st.number_input(
                        f"Box on Lap (Stop {s_idx + 1})",
                        min_value=1,
                        max_value=circuit.total_laps - 1,
                        value=default_lap,
                        step=1,
                        key=f"custom_lap_{selected_key}_{s_idx}",
                    )
                    p_comp = st.selectbox(
                        f"Fit Compound (Stop {s_idx + 1})",
                        options=list(DRY_COMPOUNDS),
                        index=default_comp_idx,
                        format_func=lambda c: f"{COMPOUNDS[c].name} ({COMPOUNDS[c].code})",
                        key=f"custom_comp_{selected_key}_{s_idx}",
                    )
                    custom_stops_input.append((int(p_lap), p_comp))

        custom_result = build_custom_strategy_plan(
            circuit=circuit,
            start_compound=custom_start_comp,
            stops=custom_stops_input,
            driver_pace_offset_sec=driver_pace_offset,
        )
        base_player_plan = custom_result.plan

        if custom_result.is_fia_legal_dry:
            st.success(f"✅ FIA 2-Compound Dry Rule Satisfied — Active Plan: **{custom_result.stint_summary}**")
        else:
            for w in custom_result.warnings:
                st.warning(f"⚠️ {w}")

elif strategy_mode == "🏁 Real Grand Prix Result Strategy":
    with st.expander("🏁 Real Grand Prix Classification & Pit-Wall Stints (FastF1)", expanded=True):
        if actual_drivers:
            drv_idx = st.selectbox(
                f"Select Driver Classification ({circuit_actuals.get('source', 'FastF1 Race')})",
                options=list(range(len(actual_drivers))),
                format_func=lambda i: actual_drivers[i].display_label,
                key=f"actual_drv_{selected_key}",
            )
            selected_actual_driver = actual_drivers[drv_idx]
            custom_result = build_actual_driver_plan(
                circuit=circuit,
                driver_strategy=selected_actual_driver,
                driver_pace_offset_sec=driver_pace_offset,
            )
            base_player_plan = custom_result.plan

            m1, m2, m3, m4 = st.columns(4)
            pos_str = f"P{selected_actual_driver.position}" if selected_actual_driver.position < 90 else "DNF"
            grid_str = f"P{selected_actual_driver.grid_position}" if selected_actual_driver.grid_position > 0 else "Pit/NA"
            m1.metric("Driver & Team", f"{selected_actual_driver.driver_name}", f"{selected_actual_driver.team}")
            m2.metric("Finish / Grid", f"{pos_str} (Started {grid_str})", selected_actual_driver.status)
            m3.metric("Laps Completed", f"{selected_actual_driver.laps_completed} / {circuit.total_laps}")
            m4.metric("Executed Pit Stops", f"{selected_actual_driver.num_stops} Stop(s)", selected_actual_driver.strategy_summary)

            for w in custom_result.warnings:
                st.info(f"ℹ️ {w}")
        else:
            st.warning(
                "No real race classification is cached for this round yet. "
                "Run `python data/calibrate_fastf1.py` after the Grand Prix completes."
            )

# --- APPLY LIVE CHAOS SCENARIO ---
sc_laps = list(range(chaos_lap, min(circuit.total_laps + 1, chaos_lap + 4))) if inject_sc else []
wetness_profile = [0.0] * circuit.total_laps
if inject_rain:
    for l in range(chaos_lap - 1, min(circuit.total_laps, chaos_lap + 8)):
        progress = (l - (chaos_lap - 1)) / 7.0
        wetness_profile[l] = 0.45 * float(np.sin(np.pi * progress))

active_rival_plan = rival_plan
if inject_undercut:
    early_lap = max(3, chaos_lap - 1)
    active_rival_plan = CarStrategyPlan(
        name=f"Rival Early Undercut (M→H) [L{early_lap}]",
        start_compound="MEDIUM",
        pit_stops={early_lap: "HARD"},
    )

# Mid-race dynamic re-optimization if Safety Car is deployed at chaos_lap (Auto DP mode)
active_player_plan = base_player_plan
if inject_sc and strategy_mode == "🤖 Bellman DP Optimal (Auto)":
    curr_comp = base_player_plan.start_compound
    curr_age = 0
    used_two = False
    kept_stops = {}
    for l in range(1, chaos_lap):
        if l in base_player_plan.pit_stops:
            nxt = base_player_plan.pit_stops[l]
            if nxt != curr_comp:
                used_two = True
            curr_comp = nxt
            curr_age = 0
            kept_stops[l] = nxt
        else:
            curr_age += 1

    reopt = solve_optimal_strategy_dp(
        circuit=circuit,
        start_lap=chaos_lap,
        current_compound=curr_comp,
        current_tire_age=curr_age,
        used_different_compound=used_two,
        is_safety_car_now=True,
        max_stops=max_stops,
        driver_pace_offset_sec=driver_pace_offset,
    )
    merged_stops = {**kept_stops, **reopt.plan.pit_stops}
    codes = [COMPOUNDS[base_player_plan.start_compound].code] + [
        COMPOUNDS[merged_stops[l]].code for l in sorted(merged_stops)
    ]
    laps_label = ", ".join(f"L{l}" for l in sorted(merged_stops))
    active_player_plan = CarStrategyPlan(
        name=f"Live Re-Opt SC: {len(merged_stops)}-Stop ({'→'.join(codes)}) [{laps_label}]",
        start_compound=base_player_plan.start_compound,
        pit_stops=merged_stops,
        driver_pace_offset_sec=driver_pace_offset,
    )

# Deterministic head-to-head race trace & Monte Carlo rollout
det_trace = simulate_race_trace(
    circuit=circuit,
    player_plan=active_player_plan,
    rival_plan=active_rival_plan,
    safety_car_laps=sc_laps,
    wetness_by_lap=wetness_profile,
)

# Clean dry baseline time for custom/actual plan so we can benchmark directly vs. DP Optimal
clean_player_trace = simulate_race_trace(
    circuit=circuit,
    player_plan=base_player_plan,
    rival_plan=rival_plan,
)
clean_player_time_sec = float(clean_player_trace["player_cum_time"][-1])
delta_vs_dp_sec = clean_player_time_sec - dp_result.expected_total_time_sec

mc_summary = run_monte_carlo_evaluation(
    circuit=circuit,
    player_plan=active_player_plan,
    rival_plan=active_rival_plan,
    n_sims=n_sims,
    enable_safety_cars=enable_sc_mc,
    enable_weather=enable_rain_mc,
)

# --- KPI METRIC STRIP ---
k1, k2, k3, k4 = st.columns(4)
clean_plan_label = (
    active_player_plan.name
    .replace("DP Optimal: ", "")
    .replace("Live Re-Opt SC: ", "")
    .replace("Custom: ", "Custom | ")
)
is_benchmark_mode = strategy_mode in ("🛠️ Custom Pit-Wall Strategy (Manual)", "🏁 Real Grand Prix Result Strategy")
if is_benchmark_mode:
    k1_delta = f"{delta_vs_dp_sec:+.2f}s vs. DP Optimal"
else:
    k1_delta = "SC Re-Optimized" if inject_sc else "Bellman Global Optimum"

k1.metric(
    label="Active Pit-Wall Strategy",
    value=clean_plan_label,
    delta=k1_delta,
    delta_color="inverse" if (is_benchmark_mode and delta_vs_dp_sec > 0.05) else "normal",
)
k2.metric(
    label="Projected Race Time",
    value=f"{det_trace['player_cum_time'][-1] / 60.0:.2f} min",
    delta=f"{det_trace['final_gap_sec']:+.2f}s vs. Rival",
)
k3.metric(
    label="Monte Carlo Win Probability",
    value=f"{mc_summary.win_probability_pct:.1f}%",
    delta=f"Median Gap: {mc_summary.p50_gap_sec:+.2f}s",
)
k4.metric(
    label="Circuit Pit-Stop Loss (Green / SC)",
    value=f"{circuit.pit_loss_sec:.1f}s / {circuit.pit_loss_sec * 0.55:.1f}s",
    delta=f"Wear Index: {circuit.tire_wear_multiplier:.2f}x",
)

st.markdown("---")

# --- ROW 1: MONTE CARLO FAN CHART & HEAD-TO-HEAD GAP ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Monte Carlo Gap Evolution (P10 – P90 Confidence Fan)")
    laps_x = det_trace["laps"]
    p10_trace = np.percentile(mc_summary.all_gap_traces, 10, axis=0)
    p50_trace = np.percentile(mc_summary.all_gap_traces, 50, axis=0)
    p90_trace = np.percentile(mc_summary.all_gap_traces, 90, axis=0)

    fig_fan = go.Figure()
    fig_fan.add_trace(
        go.Scatter(
            x=np.concatenate([laps_x, laps_x[::-1]]),
            y=np.concatenate([p90_trace, p10_trace[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 210, 190, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Stochastic Band",
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=laps_x,
            y=p50_trace,
            mode="lines",
            line=dict(color="#00D2BE", width=2.5, dash="dash"),
            name="Monte Carlo Median Gap",
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=laps_x,
            y=det_trace["gap_to_rival"],
            mode="lines",
            line=dict(color="#FF3333", width=3),
            name="Deterministic Scenario Trace",
        )
    )
    fig_fan.add_hline(y=0.0, line_dash="dot", line_color="#8B949E")
    fig_fan.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        xaxis_title="Race Lap",
        yaxis_title="Gap to Rival (+s = Player Ahead)",
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with col_right:
    st.subheader("🛞 Calibrated Tire Compound Degradation Curves")
    ages = np.arange(0, 41)
    fig_tire = go.Figure()
    for comp_key in DRY_COMPOUNDS:
        comp = COMPOUNDS[comp_key]
        deltas = [compute_tire_delta(comp_key, int(a), circuit, track_wetness=0.0) for a in ages]
        override = circuit.compound_overrides.get(comp_key, {})
        slope = override.get("deg_per_lap_sec", comp.deg_per_lap_sec * circuit.tire_wear_multiplier)
        fig_tire.add_trace(
            go.Scatter(
                x=ages,
                y=deltas,
                mode="lines",
                line=dict(color=comp.color, width=3),
                name=f"{comp.name} ({slope:.3f} s/lap)",
            )
        )
    fig_tire.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        xaxis_title="Tire Stint Age (Laps)",
        yaxis_title="Pace Delta vs. Fresh Soft (+s)",
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tire, use_container_width=True)

# --- ROW 2: DP STRATEGY LEADERBOARD (+ CUSTOM & REAL PODIUM BENCHMARKS) ---
st.subheader("🏆 Strategy Leaderboard: DP Optimum vs. Real Grand Prix Podium")

all_entries = [(label, total_sec, plan, "DP Optimal") for label, total_sec, plan in dp_result.top_alternatives]

# Add Custom Strategy if in Custom mode
if strategy_mode == "🛠️ Custom Pit-Wall Strategy (Manual)" and custom_result is not None:
    legal_tag = "Custom Strategy" if custom_result.is_fia_legal_dry else "Custom (⚠️ FIA 1-Comp)"
    all_entries.append((custom_result.stint_summary, clean_player_time_sec, custom_result.plan, legal_tag))

# Automatically benchmark real Grand Prix Podium (P1, P2, P3) + currently selected real driver
included_driver_codes = set()
podium_candidates = [d for d in actual_drivers if d.position in (1, 2, 3) and d.laps_completed >= int(circuit.total_laps * 0.85)]
if selected_actual_driver is not None and selected_actual_driver.driver_code not in {d.driver_code for d in podium_candidates}:
    podium_candidates.append(selected_actual_driver)

for drv in podium_candidates:
    if drv.driver_code in included_driver_codes:
        continue
    included_driver_codes.add(drv.driver_code)
    drv_res = build_actual_driver_plan(
        circuit=circuit,
        driver_strategy=drv,
        driver_pace_offset_sec=driver_pace_offset,
    )
    drv_trace = simulate_race_trace(
        circuit=circuit,
        player_plan=drv_res.plan,
        rival_plan=rival_plan,
    )
    drv_time_sec = float(drv_trace["player_cum_time"][-1])
    pos_badge = f"P{drv.position}" if drv.position < 90 else "DNF"
    source_label = f"Real {pos_badge} ({drv.driver_code} • {drv.team})"
    all_entries.append((drv_res.stint_summary, drv_time_sec, drv_res.plan, source_label))

all_entries.sort(key=lambda item: item[1])
fastest_time_sec = all_entries[0][1]

leaderboard_rows = []
for rank, (label, total_sec, plan, source_tag) in enumerate(all_entries, start=1):
    delta_best = total_sec - fastest_time_sec
    leaderboard_rows.append(
        {
            "Rank": f"#{rank}",
            "Source": source_tag,
            "Starting Compound": COMPOUNDS[plan.start_compound].name,
            "Strategy Sequence": label,
            "Simulated Race Time (s)": f"{total_sec:.2f}s",
            "Delta vs. Fastest": "+0.00s" if abs(delta_best) < 1e-3 else f"+{delta_best:.2f}s",
        }
    )

st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)