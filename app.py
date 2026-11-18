"""
app.py
Interactive F1 / Endurance Pit-Wall Strategy Simulator & Telemetry Dashboard.

Run locally with:
    streamlit run app.py
"""

from typing import Dict, List
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
    Circuit,
    compute_tire_delta,
    load_circuits,
    simulate_race_trace,
)

# ---------------------------------------------------------------------------
# Page Configuration & Motorsport Pit-Wall Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Pit-Wall Strategy Optimizer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #161A24;
        border: 1px solid #2A3142;
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_circuits() -> Dict[str, Circuit]:
    return load_circuits("data/circuits.json")


def format_seconds_to_race_time(total_sec: float) -> str:
    hours = int(total_sec // 3600)
    rem = total_sec % 3600
    mins = int(rem // 60)
    secs = rem % 60
    if hours > 0:
        return f"{hours}h {mins:02d}m {secs:05.2f}s"
    return f"{mins}m {secs:05.2f}s"


def build_gap_fan_chart(
    circuit: Circuit,
    det_trace: Dict[str, np.ndarray],
    mc_summary,
    player_plan: CarStrategyPlan,
    rival_plan: CarStrategyPlan,
    sc_laps: List[int],
    wetness: List[float],
) -> go.Figure:
    laps = det_trace["laps"]
    fig = go.Figure()

    # Highlight Safety Car laps as amber vertical bands
    if sc_laps:
        sc_sorted = sorted(set(sc_laps))
        start_b = sc_sorted[0]
        prev_b = sc_sorted[0]
        bands = []
        for l in sc_sorted[1:]:
            if l == prev_b + 1:
                prev_b = l
            else:
                bands.append((start_b, prev_b))
                start_b = l
                prev_b = l
        bands.append((start_b, prev_b))
        for b0, b1 in bands:
            fig.add_vrect(
                x0=b0 - 0.5,
                x1=b1 + 0.5,
                fillcolor="rgba(255, 193, 7, 0.18)",
                line_width=0,
                annotation_text="SC",
                annotation_position="top left",
                annotation_font_color="#FFC107",
            )

    # Highlight Rain laps as cyan vertical bands
    rain_laps = [i + 1 for i, w in enumerate(wetness) if w > 0.15]
    if rain_laps:
        fig.add_vrect(
            x0=min(rain_laps) - 0.5,
            x1=max(rain_laps) + 0.5,
            fillcolor="rgba(0, 180, 255, 0.14)",
            line_width=0,
            annotation_text="RAIN",
            annotation_position="bottom left",
            annotation_font_color="#38BDF8",
        )

    # Monte Carlo 10th - 90th percentile fan band
    p10_trace = np.percentile(mc_summary.all_gap_traces, 10, axis=0)
    p90_trace = np.percentile(mc_summary.all_gap_traces, 90, axis=0)
    p50_trace = np.percentile(mc_summary.all_gap_traces, 50, axis=0)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([laps, laps[::-1]]),
            y=np.concatenate([p90_trace, p10_trace[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 210, 190, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Monte Carlo 10–90% Band",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=laps,
            y=p50_trace,
            mode="lines",
            line=dict(color="#00D2BE", width=1.8, dash="dot"),
            name="Stochastic Median (P50)",
        )
    )

    # Deterministic scenario trace
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=det_trace["gap_to_rival"],
            mode="lines",
            line=dict(color="#FF3355", width=3),
            name=f"Live Scenario Gap ({player_plan.name})",
        )
    )

    # Zero-line (Track Position Reference)
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#8892B0",
        annotation_text="Rival Reference (0.0s)",
        annotation_position="bottom right",
    )

    # Plot Player pit-stop markers
    for p_lap, new_c in sorted(player_plan.pit_stops.items()):
        if 1 <= p_lap <= len(laps):
            fig.add_trace(
                go.Scatter(
                    x=[p_lap],
                    y=[det_trace["gap_to_rival"][p_lap - 1]],
                    mode="markers+text",
                    marker=dict(size=11, color=COMPOUNDS[new_c].color, line=dict(width=1.5, color="#111")),
                    text=[f"PIT→{COMPOUNDS[new_c].code}"],
                    textposition="top center",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Gap to Rival vs. Lap — {circuit.name} (+ = Player Ahead)",
        xaxis_title="Lap Number",
        yaxis_title="Gap to Rival (seconds)",
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_lap_time_chart(
    det_trace: Dict[str, np.ndarray],
    player_plan: CarStrategyPlan,
    rival_plan: CarStrategyPlan,
) -> go.Figure:
    laps = det_trace["laps"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=laps,
            y=det_trace["player_lap_times"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=[COMPOUNDS[c].color for c in det_trace["player_compounds"]],
            ),
            line=dict(color="#FF3355", width=2),
            name=f"Player: {player_plan.name}",
            customdata=np.stack([det_trace["player_compounds"], det_trace["player_tire_ages"]], axis=-1),
            hovertemplate="Lap %{x}<br>Time: %{y:.2f}s<br>Tire: %{customdata[0]} (Age %{customdata[1]})<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=laps,
            y=det_trace["rival_lap_times"],
            mode="lines",
            line=dict(color="#64B5F6", width=2, dash="dash"),
            name=f"Rival: {rival_plan.name}",
            customdata=np.stack([det_trace["rival_compounds"], det_trace["rival_tire_ages"]], axis=-1),
            hovertemplate="Lap %{x}<br>Time: %{y:.2f}s<br>Tire: %{customdata[0]} (Age %{customdata[1]})<extra></extra>",
        )
    )

    fig.update_layout(
        title="Lap-by-Lap Pace & Stint Degradation Profile (Marker Color = Tire Compound)",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_tire_cliff_chart(circuit: Circuit) -> go.Figure:
    ages = np.arange(0, min(46, circuit.total_laps))
    fig = go.Figure()
    for c_key in DRY_COMPOUNDS:
        comp = COMPOUNDS[c_key]
        deltas = [compute_tire_delta(c_key, int(a), circuit, track_wetness=0.0) for a in ages]
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=deltas,
                mode="lines",
                line=dict(color=comp.color, width=3),
                name=f"{comp.name} (Cliff ~L{int(comp.cliff_lap / (circuit.tire_wear_multiplier ** 0.5))})",
            )
        )
    fig.update_layout(
        title=f"Compound Degradation & Thermal Cliff Curves ({circuit.name})",
        xaxis_title="Stint Tire Age (Laps)",
        yaxis_title="Pace Loss vs. Fresh Soft (seconds/lap)",
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------------------------
def main():
    circuits = get_circuits()

    # Initialize session state for Chaos Buttons
    if "sc_active" not in st.session_state:
        st.session_state.sc_active = False
    if "sc_lap" not in st.session_state:
        st.session_state.sc_lap = 18
    if "rival_undercut" not in st.session_state:
        st.session_state.rival_undercut = False
    if "rain_active" not in st.session_state:
        st.session_state.rain_active = False

    # Sidebar: Circuit & Solver Controls
    with st.sidebar:
        st.title("🏎️ Pit-Wall Controls")
        circuit_key = st.selectbox(
            "Select Grand Prix Circuit",
            options=list(circuits.keys()),
            format_func=lambda k: f"{circuits[k].name} ({circuits[k].total_laps} Laps)",
            index=1,  # Default: Silverstone
        )
        circuit = circuits[circuit_key]

        st.caption(
            f"**Pit Loss:** {circuit.pit_loss_sec:.1f}s | "
            f"**Wear Mult:** {circuit.tire_wear_multiplier:.2f}x | "
            f"**Overtake Diff:** {circuit.overtaking_difficulty:.2f}"
        )

        st.divider()
        st.subheader("Strategy Configuration")
        strategy_mode = st.radio(
            "Player Strategy Mode",
            options=["DP Global Optimal (Auto)", "Custom Manual Strategy"],
            index=0,
        )

        dp_result = solve_optimal_strategy_dp(circuit)

        if strategy_mode == "DP Global Optimal (Auto)":
            player_plan = dp_result.plan
            st.success(f"**Locked Strategy:** {dp_result.stint_summary}")
        else:
            start_c = st.selectbox("Starting Compound", options=list(DRY_COMPOUNDS), index=1)
            n_stops = st.slider("Number of Pit Stops", min_value=1, max_value=3, value=1)
            custom_stops: Dict[int, str] = {}
            for s_idx in range(n_stops):
                default_l = min(circuit.total_laps - 2, int(circuit.total_laps * (s_idx + 1) / (n_stops + 1)))
                col1, col2 = st.columns(2)
                with col1:
                    p_l = st.number_input(
                        f"Stop {s_idx + 1} Lap",
                        min_value=2,
                        max_value=circuit.total_laps - 1,
                        value=default_l,
                        key=f"pit_l_{s_idx}",
                    )
                with col2:
                    p_c = st.selectbox(
                        f"Compound {s_idx + 1}",
                        options=list(DRY_COMPOUNDS),
                        index=2 if s_idx == 0 else 1,
                        key=f"pit_c_{s_idx}",
                    )
                custom_stops[int(p_l)] = p_c
            player_plan = CarStrategyPlan(
                name=f"Custom ({COMPOUNDS[start_c].code}→"
                + "→".join(COMPOUNDS[custom_stops[k]].code for k in sorted(custom_stops))
                + ")",
                start_compound=start_c,
                pit_stops=custom_stops,
            )

        rival_options = generate_standard_rival_strategies(circuit)
        rival_idx = st.selectbox(
            "Benchmark Rival Strategy",
            options=range(len(rival_options)),
            format_func=lambda i: rival_options[i].name,
            index=0,
        )
        rival_plan = rival_options[rival_idx]

        st.divider()
        st.subheader("Monte Carlo Risk Engine")
        n_sims = st.slider("Stochastic Rollouts (N)", min_value=100, max_value=800, value=300, step=50)
        mc_sc = st.checkbox("Random Safety Cars in MC", value=True)
        mc_rain = st.checkbox("Random Rain Cells in MC", value=True)

    # Header
    st.title("🏁 F1 / Endurance Pit-Wall Strategy Simulator")
    st.caption(
        f"**{circuit.name}** ({circuit.country}) — {circuit.total_laps} Laps | "
        f"Base Pace: {circuit.base_lap_time_sec:.1f}s | Fuel Load: {circuit.fuel_load_kg:.0f} kg"
    )

    # -----------------------------------------------------------------------
    # Live Pit-Wall "Chaos Buttons" Row
    # -----------------------------------------------------------------------
    st.markdown("#### ⚡ Live Race Control & Chaos Triggers")
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1.3, 1.1, 1.1, 0.8])

    with c_btn1:
        sc_trigger_lap = st.slider(
            "Safety Car / Re-Solve Lap",
            min_value=4,
            max_value=max(5, circuit.total_laps - 5),
            value=min(18, circuit.total_laps // 2),
        )
    with c_btn2:
        st.write("")
        if st.button("🟡 Toggle Safety Car Window", use_container_width=True):
            st.session_state.sc_active = not st.session_state.sc_active
    with c_btn3:
        st.write("")
        if st.button("🔴 Toggle Rival Early Undercut", use_container_width=True):
            st.session_state.rival_undercut = not st.session_state.rival_undercut
    with c_btn4:
        st.write("")
        if st.button("🌧️ Toggle Rain", use_container_width=True):
            st.session_state.rain_active = not st.session_state.rain_active

    # Construct active race perturbation vectors
    active_sc_laps: List[int] = (
        [sc_trigger_lap + d for d in range(4) if sc_trigger_lap + d <= circuit.total_laps]
        if st.session_state.sc_active
        else []
    )

    active_wetness: List[float] = [0.0] * circuit.total_laps
    if st.session_state.rain_active:
        r_start = max(5, circuit.total_laps // 2)
        for idx in range(r_start - 1, min(circuit.total_laps, r_start + 11)):
            prog = (idx - (r_start - 1)) / 11.0
            active_wetness[idx] = 0.48 * float(np.sin(np.pi * prog))

    effective_rival = rival_plan
    if st.session_state.rival_undercut and rival_plan.pit_stops:
        shifted_stops = {max(4, lap - 4): comp for lap, comp in rival_plan.pit_stops.items()}
        effective_rival = CarStrategyPlan(
            name=f"{rival_plan.name} [UNDERCUT -4L]",
            start_compound=rival_plan.start_compound,
            pit_stops=shifted_stops,
        )

    # Mid-Race Dynamic Re-Optimization Banner when Safety Car is active
    if st.session_state.sc_active:
        # Determine player's tire state entering `sc_trigger_lap`
        curr_c = player_plan.start_compound
        curr_age = 0
        used_two = False
        for l in range(1, sc_trigger_lap):
            if l in player_plan.pit_stops:
                new_c = player_plan.pit_stops[l]
                if new_c != curr_c:
                    used_two = True
                curr_c = new_c
                curr_age = 0
            else:
                curr_age += 1

        sc_reopt = solve_optimal_strategy_dp(
            circuit=circuit,
            start_lap=sc_trigger_lap,
            current_compound=curr_c,
            current_tire_age=curr_age,
            used_different_compound=used_two,
            is_safety_car_now=True,
        )
        pit_now = sc_trigger_lap in sc_reopt.plan.pit_stops
        if pit_now:
            rec_comp = sc_reopt.plan.pit_stops[sc_trigger_lap]
            st.warning(
                f"🟡 **SAFETY CAR DEPLOYED ON LAP {sc_trigger_lap}!** — "
                f"**PIT WALL CALL: BOX THIS LAP FOR {rec_comp}!** "
                f"(Exploits 45% SC pit-loss discount; optimal remaining plan: {sc_reopt.stint_summary})"
            )
            if st.checkbox("Apply Live SC Pit-Wall Call to Simulation", value=True):
                updated_stops = {k: v for k, v in player_plan.pit_stops.items() if k < sc_trigger_lap}
                updated_stops.update(sc_reopt.plan.pit_stops)
                player_plan = CarStrategyPlan(
                    name=f"SC Re-Opt ({sc_reopt.stint_summary})",
                    start_compound=player_plan.start_compound,
                    pit_stops=updated_stops,
                )
        else:
            st.info(
                f"🟡 **SAFETY CAR DEPLOYED ON LAP {sc_trigger_lap}!** — "
                f"**PIT WALL CALL: STAY OUT** (Tires age {curr_age}L; pitting now costs track position)."
            )

    # Adapt plans if rain toggle is active
    sim_player = player_plan
    sim_rival = effective_rival
    if st.session_state.rain_active:
        from engine.optimizer import _adapt_plan_for_rain
        sim_player = _adapt_plan_for_rain(player_plan, active_wetness)
        sim_rival = _adapt_plan_for_rain(effective_rival, active_wetness)

    # Run Deterministic Trace + Monte Carlo Rollout
    det_trace = simulate_race_trace(
        circuit=circuit,
        player_plan=sim_player,
        rival_plan=sim_rival,
        safety_car_laps=active_sc_laps,
        wetness_by_lap=active_wetness,
        stochastic=False,
    )

    mc_summary = run_monte_carlo_evaluation(
        circuit=circuit,
        player_plan=player_plan,
        rival_plan=effective_rival,
        n_sims=n_sims,
        enable_safety_cars=mc_sc,
        enable_weather=mc_rain,
    )

    # -----------------------------------------------------------------------
    # Top KPI Telemetry Strip
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        label="Active Player Strategy",
        value=player_plan.name.replace("DP Optimal: ", ""),
        delta=f"vs {effective_rival.name.split('[')[0].strip()}",
    )
    k2.metric(
        label="Live Scenario Finish Gap",
        value=f"{det_trace['final_gap_sec']:+.2f}s",
        delta="Ahead of Rival" if det_trace["final_gap_sec"] > 0 else "Behind Rival",
    )
    k3.metric(
        label=f"Monte Carlo Win Prob ({n_sims} sims)",
        value=f"{mc_summary.win_probability_pct:.1f}%",
        delta=f"Mean Gap: {mc_summary.mean_final_gap_sec:+.2f}s (±{mc_summary.std_final_gap_sec:.1f}s)",
    )
    k4.metric(
        label="Projected Total Race Time",
        value=format_seconds_to_race_time(float(det_trace["player_cum_time"][-1])),
        delta=f"DP Baseline: {format_seconds_to_race_time(dp_result.expected_total_time_sec)}",
        delta_color="off",
    )

    # -----------------------------------------------------------------------
    # Interactive Telemetry Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Gap-to-Rival & Monte Carlo Fan",
            "⏱️ Lap Pace & Tire Degradation Cliffs",
            "🏆 DP Strategy Matrix & Risk Histogram",
        ]
    )

    with tab1:
        st.plotly_chart(
            build_gap_fan_chart(
                circuit=circuit,
                det_trace=det_trace,
                mc_summary=mc_summary,
                player_plan=sim_player,
                rival_plan=sim_rival,
                sc_laps=active_sc_laps,
                wetness=active_wetness,
            ),
            use_container_width=True,
        )

    with tab2:
        col_left, col_right = st.columns([1.4, 1.0])
        with col_left:
            st.plotly_chart(
                build_lap_time_chart(det_trace, sim_player, sim_rival),
                use_container_width=True,
            )
        with col_right:
            st.plotly_chart(
                build_tire_cliff_chart(circuit),
                use_container_width=True,
            )

    with tab3:
        col_dp, col_hist = st.columns([1.0, 1.2])
        with col_dp:
            st.subheader("Dynamic Programming Starting-Compound Ranking")
            rows = []
            for rank, (label, est_t, plan_obj) in enumerate(dp_result.top_alternatives, start=1):
                rows.append(
                    {
                        "Rank": f"#{rank}",
                        "Start Tire": plan_obj.start_compound,
                        "Optimal Stint Sequence": label,
                        "Dry Race Time (s)": round(est_t, 2),
                        "Delta to Best (s)": f"+{est_t - dp_result.expected_total_time_sec:.2f}s",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Bellman backward induction solves over (Lap × Compound × Tire Age × 2-Compound FIA Rule) "
                "to guarantee global optimality."
            )

        with col_hist:
            st.subheader("Monte Carlo Finishing Gap Distribution")
            hist_fig = go.Figure()
            hist_fig.add_trace(
                go.Histogram(
                    x=mc_summary.all_gaps_sec,
                    nbinsx=35,
                    marker_color="#00D2BE",
                    opacity=0.82,
                    name="Simulated Final Gaps",
                )
            )
            hist_fig.add_vline(x=0.0, line_dash="dash", line_color="#FF3355", annotation_text="Win/Loss Threshold")
            hist_fig.update_layout(
                xaxis_title="Final Gap to Rival at Checkered Flag (seconds)",
                yaxis_title="Simulation Count",
                template="plotly_dark",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(hist_fig, use_container_width=True)


if __name__ == "__main__":
    main()