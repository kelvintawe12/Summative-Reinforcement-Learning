# Reinforcement Learning Summative Report

**Student:** Tawe Kelvin
**Project:** Waste Segregation Sorting Station — a custom Gymnasium environment trained with DQN, REINFORCE, A2C, and PPO
**Repository:** https://github.com/kelvintawe12/Summative-Reinforcement-Learning

> **How to use this scaffold.** Fill every `[[FILL: …]]` marker after Colab
> training and after running `python -m results.analysis` and
> `python -m results.generalization` locally. Insert the generated figures
> from `results/plots/` and the tables from `results/hyperparameter_tables/`.
> Target length **7–10 pages**. Keep prose tight — the rubric rewards focus,
> not volume. Export to PDF (VS Code "Markdown PDF", Pandoc, or paste into
> Google Docs) and submit the PDF to Canvas.

---

## 1. Environment Overview (≈1 page)

### 1.1 Problem statement
A single sorting station on a municipal solid waste (MSW) conveyor line. On
each timestep a new waste item arrives; the agent sees a **noisy** sensor
reading of its material composition and must route it — sort into one of four
bins, reject to landfill, hold, or trigger a costly close scan.

### 1.2 Why it is a genuine MDP (not a classification task)
Six interacting mechanics give real sequential structure (past actions change
future reward and future action availability):

1. Bin capacity with soft value decay.
2. Contamination cascades (value realized late, at ship-out).
3. Sensor noise proportional to item ambiguity.
4. Conveyor time pressure (superlinear stalling penalty).
5. Equipment jamming driven by the agent's own history.
6. Finite scan-energy budget.

### 1.3 Environment visualization
![Facility render](../assets/render_preview.png)

*Figure 1. Pygame facility dashboard: incoming item (noisy composition),
four bins with live fill/contamination and jam state, energy gauge, and
running stats.*

### 1.4 Action space (`Discrete(7)`)
| Idx | Action | Real-world meaning |
|---|---|---|
| 0–3 | `sort_organic/plastic/metal/glass` | Route item to the matching bin |
| 4 | `reject` | Divert to landfill |
| 5 | `hold` | Wait one step (item stays on belt) |
| 6 | `scan_closely` | Spend energy for a lower-noise reading |

### 1.5 Observation space (`Box(21,)`, normalized `[0,1]`)
Bin fills (4), contamination (4), jam cooldowns (4), energy (1), consecutive
holds (1), noisy item composition (5), item mass (1), steps remaining (1).

### 1.6 Reward structure
Dominated by value realized at periodic bin **ship-outs**, discounted convexly
by accumulated contamination; smaller immediate shaping on correct sorts;
penalties for overflow, jamming, forced rejection from stalling, and wasting
valuable material. Sanity check: a capacity-aware heuristic scores far above a
random policy (`tests/test_environment.py`).

### 1.7 Start state & termination
- **Start:** empty bins, full energy, one freshly spawned item.
- **Terminated:** all four bins jammed simultaneously (facility shutdown).
- **Truncated:** `max_steps` (200) reached; a final ship-out realizes value.

---

## 2. Implemented Methods (≈0.5 page)
All four algorithms train against the **same** `WasteSegregationEnv` for a
fair comparison. DQN, A2C, PPO use Stable-Baselines3; REINFORCE is implemented
from scratch (SB3 has none) with an optional learned baseline and entropy
bonus. Each algorithm is swept over **10 hyperparameter configurations**
(`training/hyperparameters.py`), 60k timesteps each.

---

## 3. Hyperparameter Experiments & Analysis (≈2–3 pages)

For each algorithm: paste the 10-row table from
`results/hyperparameter_tables/<algo>_hyperparameters.csv` and write 3–5
sentences on what the tuning revealed.

### 3.1 DQN
[[FILL: dqn_hyperparameters.csv — 10 rows]]

**Discussion.** [[FILL: effect of learning_rate, gamma, buffer_size,
exploration_fraction, target_update_interval on stability/convergence/
exploration. Reference the DQN objective curve below.]]

![DQN reward curves](../results/plots/dqn_reward_curves.png)
![DQN objective curves](../results/plots/dqn_objective_curves.png)

*Figures. DQN reward across the sweep, and the TD-loss (objective) curve.*

### 3.2 REINFORCE
[[FILL: reinforce_hyperparameters.csv — 10 rows]]

**Discussion.** [[FILL: baseline on/off variance reduction, entropy_coef vs
exploration, learning_rate sensitivity. Reference the entropy curve.]]

![REINFORCE reward curves](../results/plots/reinforce_reward_curves.png)
![REINFORCE entropy curves](../results/plots/reinforce_entropy_curves.png)

### 3.3 A2C
[[FILL: a2c_hyperparameters.csv — 10 rows]]

**Discussion.** [[FILL: n_steps, ent_coef, vf_coef, gae_lambda effects.]]

![A2C reward curves](../results/plots/a2c_reward_curves.png)
![A2C entropy curves](../results/plots/a2c_entropy_curves.png)

### 3.4 PPO
[[FILL: ppo_hyperparameters.csv — 10 rows]]

**Discussion.** [[FILL: clip_range, n_epochs, n_steps, batch_size, ent_coef
effects; PPO stability vs A2C.]]

![PPO reward curves](../results/plots/ppo_reward_curves.png)
![PPO entropy curves](../results/plots/ppo_entropy_curves.png)

---

## 4. Discussion & Analysis (≈2 pages)

### 4.1 Cross-algorithm comparison
![Convergence subplots](../results/plots/convergence_subplots.png)
![Best run per algorithm](../results/plots/algorithm_comparison.png)

**Discussion.** [[FILL: which algorithm converged fastest / highest / most
stably. Integrate numbers from the tables. Exploration vs exploitation
evidence from entropy curves. Stability evidence from loss curves.]]

### 4.2 Generalization test
Models are trained on the default demand profile and evaluated under shifted
profiles they never saw (`high_contam`, `plastic_surge`, `uniform`).

[[FILL: generalization_results.csv table]]
![Generalization](../results/plots/generalization.png)

**Discussion.** [[FILL: which policy transferred best; how large the drop was
under distribution shift; what that says about overfitting.]]

---

## 5. System Implementation & Deployment (≈0.5 page)
- **Visualization:** Pygame facility dashboard (`environment/rendering.py`).
- **Web/mobile integration:** the environment serializes to JSON
  (`WasteSegregationEnv.to_json()`) and is exposed over a REST API
  (`serve.py`, FastAPI) — reset/step/predict endpoints return the full
  facility state, so a browser or app can render it with no Python
  dependency. Run: `uv run uvicorn serve:app --reload` → `/docs`.
- **Best agent demo:** `uv run main.py` auto-selects and runs the best model.

---

## 6. Conclusion (≈0.25 page)
[[FILL: best algorithm and why; one concrete limitation; one line of future
work (e.g. multi-item queue, seasonal contamination shift).]]

---

## Appendix — Reproducibility
```bash
uv sync
uv run pytest tests/ -v          # environment + dynamics tests
# Train (locally or via notebooks/ on Colab):
uv run python -m training.dqn_training --run all
uv run python -m training.reinforce_training --run all
uv run python -m training.a2c_training --run all
uv run python -m training.ppo_training --run all
# Analysis + generalization:
uv run python -m results.analysis
uv run python -m results.generalization
uv run python main.py            # watch the best agent
```
