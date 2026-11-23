# 🏎️ F1 / Endurance Pit-Wall Strategy Simulator (v2.4 — 2026 Season Edition)

![view](/Users/elliottparker/Library/CloudStorage/OneDrive-Personal/Documents/Projects/f1-pitwall-optimizer/f1-pitwall-optimizer/view.png)

An interactive motorsport race-strategy optimizer, real-race telemetry benchmarker, and stochastic pit-wall decision engine built in Python. Combines **dual-source OpenF1 & FastF1 telemetry calibration** across all **23 circuits of the 2026 FIA Formula 1 World Championship** with **exact Backward Dynamic Programming (Bellman optimality)** and **Monte Carlo stochastic race simulations**.

🌐 **Live Interactive App:** [Launch Pit-Wall Simulator on Streamlit Cloud](https://f1-pitwall-optimizer.streamlit.app)

---

## ✨ Key Capabilities

### 1. Dual-Source OpenF1 & FastF1 Telemetry ETL Pipeline (`data/calibrate_fastf1.py`)
- **Official FIA Race Classifications & Pit Stints (OpenF1 REST API):** Queries `/sessions`, `/session_result`, `/drivers`, `/stints`, and `/laps` with HTTP 429 exponential backoff and `circuit_short_name` / `country_name` fallback resolution across all **23 rounds of the 2026 F1 calendar** (R01 Australia through R23 Abu Dhabi, including Madring and Sepang). Exports `data/actual_strategies.json`.
- **Empirical Lap & Tire Calibration (FastF1):** Filters out Lap 1 standing starts, pit-in/pit-out laps, Safety Car / VSC periods (`TrackStatus != '1'`), and 107% outliers using local disk caching (`data/fastf1_cache/`).
- **Physics Parameter Extraction:** Removes fuel-mass burn effects to isolate pure **clean-air base lap pace**, **empirical pit-lane time loss**, **Safety Car deployment probability**, and **per-compound linear degradation slopes** (`SOFT`, `MEDIUM`, `HARD`) into `data/circuits.json` and `data/compound_profiles.json`.

### 2. Lap-by-Lap Vehicle & Tire Physics Engine (`engine/physics.py`)
- **Fuel Mass Burn:** Linear fuel burn model reducing car mass and lap time over the race distance.
- **Compound Degradation & Thermal Cliff:** Circuit-specific empirical pace offsets and linear rubber wear combined with a non-linear quadratic performance cliff beyond each compound's thermal life:
  $$\Delta t_{\text{tire}}(c, a, w) = \Delta t_{0,c} + \alpha_c \cdot a + \beta_c \cdot \max(0, a - a_{\text{cliff},c})^2 + \Delta t_{\text{wet}}(c, w)$$
- **Wake Aerodynamics & Overtaking Thresholds:** Models dirty-air downforce loss when trailing within `1.0s` unless net pace advantage overcomes circuit-specific overtaking difficulty.
- **Safety Car Neutralization:** Compresses field gaps and discounts pit-stop delta by ~45% during Safety Car windows.

### 3. Exact Backward Dynamic Programming & Strategy Benchmarking (`engine/optimizer.py`)
- **Bellman Optimality Solver:** Solves the finite-horizon Bellman optimality equations across state space `(lap, compound, tire_age, used_two_compounds, stops_used)` with exact stop-count partitioning and commutative stint deduplication to find the globally optimal pit-stop lap(s) and compound sequence.
- **FIA Sporting Rules Enforcement:** Strictly enforces the sporting regulation requiring at least **two distinct dry compounds** in a dry race (with automatic weather exemption when `INTER` or `WET` tires are fitted).
- **Live Mid-Race Re-Optimization:** Instantaneously re-solves optimal pit windows from any current lap, compound, and tire age when a Safety Car deploys mid-race.
- **Custom & Real Grand Prix Strategy Builders:** Validates manual pit-wall strategies (`build_custom_strategy_plan`) and loads actual FIA P1–P20 Grand Prix driver strategies (`build_actual_driver_plan`) for direct head-to-head delta benchmarking against the Bellman DP global optimum.

### 4. Monte Carlo Risk Engine, Interactive Pit-Wall UI & Engineering Manual (`app.py`)
- **Monte Carlo Stochastic Rollouts:** Stress-tests candidate strategies across 100–600 stochastic race futures featuring random Safety Car windows, passing rain cells (with automatic `Dry ↔ Intermediate ↔ Full Wet` crossover rules), and Gaussian lap-time variance.
- **Dark-Mode Pit-Wall Command Center:** Built with **Streamlit + Plotly**, featuring 3 strategy modes (`Bellman DP Optimal`, `Custom Pit-Wall Strategy`, `Real Grand Prix Result Strategy`), live chaos toggles, P10–P90 confidence fan charts, calibrated compound degradation curves, and a unified Strategy Leaderboard comparing DP optima against real Grand Prix podium finishers (`P1–P3`).
- **Built-In Physics & Rules Manual + `?` Tooltips:** Includes a dedicated 6-tab **Engineering, Physics & FIA Rules Manual** view inside the app alongside native `?` hover tooltips on every metric, slider, chart, and leaderboard column.

---

## 📁 Project Architecture

```text
f1-pitwall-optimizer/
├── data/
│   ├── calibrate_fastf1.py     # OpenF1 + FastF1 ETL, empirical calibration & real GP results pipeline
│   ├── circuits.json           # Calibrated 2026 circuit boundary conditions (R01–R23)
│   ├── compound_profiles.json  # Circuit-specific empirical Soft/Medium/Hard wear fits
│   └── actual_strategies.json  # Official FIA driver classifications & executed pit-stop sequences
├── engine/
│   ├── __init__.py
│   ├── physics.py              # Fuel, tire cliff, wet crossover, dirty air & race trace physics
│   └── optimizer.py            # Backward DP solver, Custom/Actual builders, mid-race SC & Monte Carlo engine
├── app.py                      # Streamlit + Plotly pit-wall dashboard & Engineering/Physics Manual
├── requirements.txt            # numpy, scipy, pandas, plotly, streamlit, fastf1
├── LICENSE                     # MIT License
└── README.md

🚀 Quickstart (Local Setup)

1. Clone & Create Virtual Environment
git clone [https://github.com/](https://github.com/)elli0tt-c/f1-pitwall-optimizer.git
cd f1-pitwall-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run OpenF1 / FastF1 Telemetry Calibration (Optional — Pre-Calibrated JSONs Included)
python data/calibrate_fastf1.py

3. Launch the Interactive Pit-Wall Dashboard
streamlit run app.py

🛠️ Tech Stack
Telemetry & Data Engineering: fastf1, OpenF1 REST API, pandas, numpy
Optimization & Simulation: Exact Backward Dynamic Programming (NumPy), Monte Carlo Stochastic Rollouts
Interactive Visualization & Deployment: streamlit, plotly, Streamlit Community Cloud

📜 License
Released under the MIT License.