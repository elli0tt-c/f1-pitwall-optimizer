# 🏎️ F1 / Endurance Pit-Wall Strategy Simulator

An interactive motorsport race-strategy optimization engine and pit-wall telemetry dashboard built in Python using **Dynamic Programming**, **Monte Carlo stochastic rollouts**, and **lap-by-lap vehicle physics**.

---

## 🎯 Engineering Overview

In Formula 1 and Endurance racing, race strategy is a constrained sequential decision problem under uncertainty. Pit walls must balance deterministic tire degradation and fuel-burn physics against stochastic race disruptions (Safety Cars, passing rain cells, and rival undercuts).

This simulator combines three core modules:

1. **Lap-by-Lap Vehicle & Track Physics (`engine/physics.py`)**
   - **Fuel Mass Reduction:** Linear fuel burn decreasing vehicle mass and improving lap time via $\Delta t_{\text{fuel}}(k) = m_{\text{fuel}}(k) \cdot c_{\text{fuel}}$.
   - **Non-Linear Tire Degradation & Thermal Cliff:** Compound-specific linear rubber wear combined with a quadratic performance cliff once stint age $a$ exceeds the thermal/wear threshold $a_{\text{cliff}}$:
     $$\Delta t_{\text{tire}}(c, a) = \Delta t_{0,c} + \alpha_c \mu_{\text{track}} a + \beta_c \max(0, a - a_{\text{cliff}})^2$$
   - **Wet-Weather Compound Crossover:** Continuous track-wetness modeling across Slicks, Intermediates, and Full Wets.
   - **Wake Aerodynamics:** Dirty-air downforce penalty when trailing within $1.0\text{ s}$ of a rival car without sufficient pace delta to overcome circuit overtaking difficulty.

2. **Bellman Dynamic Programming & Monte Carlo Solver (`engine/optimizer.py`)**
   - **Exact Backward Dynamic Programming:** Solves for the globally optimal pit-stop lap(s) and compound sequence over state space $(\text{Lap}, \text{Compound}, \text{Tire Age}, \text{2-Compound FIA Rule}, \text{Stops})$ in $<15\text{ ms}$.
   - **Mid-Race Dynamic Re-Optimization:** Re-solves optimal remaining stints from any live lap (e.g., evaluating whether to box immediately under a Safety Car's $45\%$ pit-loss discount).
   - **Monte Carlo Rollout Engine:** Stress-tests strategies across hundreds of randomized race futures with Poisson-like Safety Car windows, passing rain cells, and stochastic lap-time perturbations to compute **Win Probability (%)** and **10th–90th percentile confidence bands**.

3. **Interactive Pit-Wall Dashboard (`app.py`)**
   - Real-time **Streamlit + Plotly** command center featuring live gap-to-rival fan charts, compound degradation curves, DP strategy leaderboards, and interactive **"Chaos Buttons"** (Deploy Safety Car, Rival Early Undercut, Passing Rain Shower).

---

## 📂 Project Structure

```text
f1-pitwall-optimizer/
├── app.py                  # Streamlit + Plotly interactive pit-wall dashboard
├── requirements.txt        # Python dependencies (NumPy, SciPy, Pandas, Plotly, Streamlit)
├── data/
│   └── circuits.json       # Circuit physics & operational presets (Monza, Silverstone, Suzuka, Monaco, Bahrain, Spa)
└── engine/
    ├── __init__.py
    ├── physics.py          # Fuel burn, tire cliff, dirty-air aero, and race trace simulation
    └── optimizer.py        # Backward Dynamic Programming solver + Monte Carlo rollout engine

🚀 Quickstart (Run Locally)
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/f1-pitwall-optimizer.git](https://github.com/YOUR_USERNAME/f1-pitwall-optimizer.git)
cd f1-pitwall-optimizer

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the interactive Pit-Wall Dashboard
streamlit run app.py

🛠️ Tech Stack
Core Computation: Python 3, NumPy, SciPy, Pandas
Visualization & UI: Plotly, Streamlit