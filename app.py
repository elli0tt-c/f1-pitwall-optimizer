"""
app.py
F1 / Endurance Pit-Wall Strategy Simulator (v2.0 — 2026 Season FastF1 Telemetry Edition)

Interactive Streamlit + Plotly pit-wall command center featuring:
- All 23 circuits of the 2026 FIA Formula 1 World Championship calibrated via FastF1
- Exact Backward Dynamic Programming (Bellman optimality) solver enforcing the 2-compound rule
- Live mid-race chaos controls (Safety Car deployment, Rival Undercut, Passing Rain Cell)
- Monte Carlo stochastic race rollout engine with P10–P90 confidence fan chart
- Circuit-specific empirical tire compound degradation & thermal cliff curves
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.optimizer import (
    generate_standard_rival_strategies,
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


@st.cache_data
def get_all_circuits():
    return load_circuits()


circuits = get_all_circuits()
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

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Solver & Monte Carlo Settings")
max_stops = st.sidebar.slider("Max Allowed Pit Stops", min_value=1, max_value=3, value=2)
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
rival_idx = st.selectbox(
    "Select Primary Rival Strategy Benchmark",
    options=list(range(len(rival_strategies))),
    format_func=lambda i: rival_strategies[i].name,
)
rival_plan = rival_strategies[rival_idx]

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

# Mid-race dynamic re-optimization if Safety Car is deployed at chaos_lap
active_player_plan = dp_result.plan
if inject_sc:
    # Determine player's tire state entering chaos_lap
    curr_comp = dp_result.plan.start_compound
    curr_age = 0
    used_two = False
    kept_stops = {}
    for l in range(1, chaos_lap):
        if l in dp_result.plan.pit_stops:
            nxt = dp_result.plan.pit_stops[l]
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
    codes = [COMPOUNDS[dp_result.plan.start_compound].code] + [
        COMPOUNDS[merged_stops[l]].code for l in sorted(merged_stops)
    ]
    laps_label = ", ".join(f"L{l}" for l in sorted(merged_stops))
    active_player_plan = CarStrategyPlan(
        name=f"Live Re-Opt SC: {len(merged_stops)}-Stop ({'→'.join(codes)}) [{laps_label}]",
        start_compound=dp_result.plan.start_compound,
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
k1.metric(
    label="Optimal Pit-Wall Call",
    value=active_player_plan.name.replace("DP Optimal: ", "").replace("Live Re-Opt SC: ", ""),
    delta="SC Re-Optimized" if inject_sc else "Bellman Global Optimum",
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

# --- ROW 2: DP STRATEGY LEADERBOARD & CIRCUIT TELEMETRY SUMMARY ---
st.subheader("🏆 Dynamic Programming Strategy Leaderboard (2-Compound Rule Enforced)")
leaderboard_rows = []
for rank, (label, total_sec, plan) in enumerate(dp_result.top_alternatives, start=1):
    delta_best = total_sec - dp_result.expected_total_time_sec
    leaderboard_rows.append(
        {
            "Rank": f"#{rank}",
            "Starting Compound": COMPOUNDS[plan.start_compound].name,
            "Strategy Sequence": label,
            "Total Race Time (s)": f"{total_sec:.2f}s",
            "Delta vs. Optimal": "+0.00s" if rank == 1 else f"+{delta_best:.2f}s",
        }
    )

st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)