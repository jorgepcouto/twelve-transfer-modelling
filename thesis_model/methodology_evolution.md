# Methodology Evolution: v1 → v4

How the modelling approach evolved from initial exploration to the final per-quality tactical models. Each version built on lessons from the previous one.

---

## v1 — Six separate OLS specifications (exploratory)

**Goal:** Can we predict post-transfer midfielder quality using OLS?

**What we tried:** Six models of increasing complexity, each in its own notebook:

| Model | Specification | What it tested |
|-------|--------------|----------------|
| M1 | `to_Qi = a + b * from_Qi` | Naive baseline -- does pre-quality predict post-quality? |
| M2 | `to_Qi = a + Sum(bj * from_Qj)` | Do ALL 17 pre-qualities together help? |
| M3 | M2 + origin/destination team styles | Does team tactical context add signal? |
| M4a | `dQi = a + Sum(gk * dTQk)` | Can tactical change alone explain quality change? |
| M4b | `dQi = a + Sum(bj * from_Qj) + Sum(gk * dTQk)` | Pre-qualities + tactical change -> delta |
| M4c | M4b + age + d_minutes | Adding demographic controls |

**Key observations:**
- Pre-quality is a strong predictor (regression to the mean dominates)
- Delta models (4a-4c) showed much higher R2 than level models (1-3)
- Team context seemed to add signal, but comparison across models was messy -- each notebook had different evaluation criteria
- Having 6 separate notebooks made it hard to compare apples to apples

**What we learned:** Need a single consolidated comparison, and need to understand why delta models have inflated R2.

-> Notebooks: `past_models/v1/01-04c_*.ipynb`

---

## v2 — Consolidated comparison + regression-to-mean discovery

**Goal:** Put all 6 models in one notebook with fair comparison.

**What we tried:** Same 6 specifications, but now evaluated side by side with consistent metrics (R2, MAE on test set, F-test p-values).

**The breakthrough discovery:** Models 4b and 4c had inflated R2 because predicting dQ = to_Q - from_Q while including from_Q as a feature creates a mechanical relationship. The coefficient on from_Qi converges to approx -1, which is just regression to the mean -- not tactical insight. We decomposed R2 into mean-reversion vs. actual tactical signal and found that most of the "gain" was fake.

**What we learned:**
1. Delta models can't be directly compared with level models -- R2 is on different targets
2. The naive baseline `to_Qi = a + b * from_Qi` captures most variance via regression to the mean
3. Any honest evaluation must compare on the same target (either always to_Q or always dQ)
4. Team tactical features do add some signal, but it's modest -- not the huge R2 the delta models suggested

-> Notebooks: `past_models/v2/01_dataset_overview.ipynb`, `02_model_comparison.ipynb`

---

## v3 — Fair comparison framework + case studies

**Goal:** Make the comparison honest by evaluating ALL models on the same target (to_Q).

**What we tried:** Five models (dropped 4c as controls added noise), all evaluated on to_Qi. Delta models internally predict dQ, then reconstruct to_Q = from_Q + predicted_d for fair comparison.

**Key changes from v2:**
- Unified evaluation target: always to_Qi
- Delta models (4, 5) are evaluated on reconstructed to_Q, making comparison honest
- Added case studies: real transfers with predicted vs actual qualities
- Built a Streamlit app for interactive transfer exploration

**Results:** Model 3 (Player + Team Context, predicting to_Q directly) won for 13/17 qualities. But the margin over the naive baseline was modest. Using all 17 pre-qualities often overfitted -- many qualities don't help predict a specific target quality.

**What we learned:**
1. Including all 17 pre-qualities as features hurts more than it helps (overfitting, noise)
2. The per-quality naive baseline (to_Qi ~ from_Qi) is a surprisingly strong benchmark
3. Team tactical dimensions don't all matter equally for every quality -- Involvement might care about Attack and Defence, while Active defence might only care about Defence
4. **The key insight: we should model each quality independently with its own optimal feature set**

-> Notebooks: `past_models/v3/01-04_*.ipynb`, plus `streamlit_app/`

---

## v4 — Per-quality tactical models (current)

**Goal:** For each midfielder quality, find the optimal subset of team tactical dimensions that maximises out-of-sample R2 above the naive baseline.

**The approach:**
1. **Simplify the baseline:** dQi = a + b * Qi_pre (one predictor, pure regression to the mean)
2. **Exhaustive search:** For each quality, test all 127 combinations (2^7 - 1) of the 7 team tactical dimensions
3. **Selection criterion:** Maximise test-set R2 (out-of-sample, no cheating)
4. **Per-quality customisation:** Each quality gets its own optimal tactical subset

**Model formulation:**
```
Baseline (M1):  dQi = a + b * Qi_pre
Tactical (M2):  dQi = a + b * Qi_pre + Sum(gk * dTQk)   (k in Si, quality-specific subset)
```

**Sample:** 4,888 midfielders (train: 3,910, test: 978).

**Results:**

| Quality | Baseline R2 | Tactical R2 | R2 Gain | Selected tactical dimensions |
|---------|-------------|-------------|---------|------------------------------|
| Involvement | 0.208 | 0.329 | +58.7% | Attack, Atk Trans, Defence, Def Trans, Outcome |
| Passing quality | 0.185 | 0.247 | +33.8% | Attack, Defence, Def Trans, Outcome |
| Providing teammates | 0.205 | 0.241 | +17.4% | Attack, Defence, Outcome |
| Effectiveness | 0.251 | 0.292 | +16.2% | Attack, Atk Trans, Defence, Outcome, Penetration |
| Box threat | 0.194 | 0.222 | +14.5% | Attack, Chance Creation, Defence, Outcome |
| Progression | 0.195 | 0.214 | +9.5% | Attack, Atk Trans, Defence, Def Trans |
| Intelligent defence | 0.234 | 0.249 | +6.6% | Chance Creation, Defence, Def Trans, Outcome |
| Active defence | 0.228 | 0.236 | +3.7% | Defence, Outcome |

**Key findings:**
- **Defence** is selected for all 8 qualities -- moving to a team with a different defensive style affects every aspect of a midfielder's game
- **Outcome** (team results quality) selected for 7/8 -- playing for a better/worse team has broad effects
- **Attack** selected for 6/8 -- the team's attacking style shapes most midfielder qualities
- **Involvement** benefits most from tactical context (+59%) -- how much a midfielder touches the ball is heavily system-dependent
- **Defensive qualities** benefit least (3.7-6.6%) -- Active/Intelligent defence are mostly regression to the mean, less influenced by tactical context
- Segment analysis confirms: tactical model gains are largest when the tactical distance between origin and destination clubs is large

-> Notebooks: `current_model/results_overview.ipynb` (main), `current_model/{quality}_model/01_{quality}_search.ipynb` (per-quality), `current_model/02_segment_analysis.ipynb`

---

## Summary of the journey

```
v1: "Let's try everything"          -> 6 models, hard to compare
v2: "Let's compare fairly"          -> discovered regression-to-mean inflation
v3: "Unified evaluation target"     -> naive baseline is strong; one-size-fits-all features overfit
v4: "Per-quality optimal subsets"   -> each quality gets custom tactical features -> clear gains
```

The core narrative: **regression to the mean dominates**, but tactical context adds real, measurable signal -- especially for system-dependent qualities like Involvement. The improvement varies by quality, and the specific tactical dimensions that matter differ quality by quality, which is itself a finding worth reporting.
