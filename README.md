# Twelve Transfer Modelling

Master thesis: predicting how football players' performance qualities change after within-league transfers, using Twelve Quality Scores and team tactical styles.

## Data sources

- **Twelve Football (Wyscout)** — player performance metrics, team stats, quality scores
- **Transfermarkt** — transfer records, fees, market values

Raw data stored externally in `../thesis_data/` (not committed).

## Repository structure

```
twelve-transfer-modelling/
├── thesis_model/
│   ├── preprocessing/              # Data pipeline (01–08, with sub-notebooks 03_1, 08_1)
│   ├── thesis_models_v1/           # 6 separate OLS notebooks (midfielders)
│   ├── thesis_models_v2/           # Consolidated, regression-to-mean analysis
│   ├── thesis_models_v3/           # Fair comparison: 5 models, case studies
│   │   └── streamlit_app/          # Interactive transfer explorer (Streamlit)
│   └── thesis_models_v4/           # Per-quality models: naive vs full
│       ├── passing_quality_model/
│       ├── progression_model/
│       ├── effectiveness_model/
│       ├── providing_teammates_model/
│       ├── involvement_model/
│       ├── active_defence_model/
│       ├── box_threat_model/
│       ├── intelligent_defence_model/
│       ├── summary.ipynb           # Cross-quality comparison
│       └── 02_segment_analysis.ipynb # Tactical gain by transfer segment
├── practice_model/                 # Earlier exploratory iterations (v1, v2)
├── twelve_qualities/               # Quality definitions
│   ├── player_qualities.csv
│   ├── team_qualities.txt
│   └── team.py                     # Team quality computation helpers
└── CLAUDE.md
```

## Data pipeline

The preprocessing pipeline (notebooks 01–08) processes a master dataset (262K transfers, 539 columns) into a modelling-ready parquet:

1. **Master** → 262K rows, 539 cols
2. **Filter** → ~18K rows (within-league, same position, 900+ min both sides)
3. **Clean columns** → remove identity/metadata, drop goalkeepers
4. **Keep core features** → Twelve Quality Scores (20×2) + team qualities (7×2) + structural
5. **Compute team qualities** → 7 tactical style dimensions per team-season

## Model iterations

### v1 — One notebook per model (6 specifications)
OLS regressions with increasing complexity. Midfielders only.

### v2 — Consolidated comparison
Discovered that delta models have inflated R² due to regression-to-mean (`from_Q` coef ≈ −1).

### v3 — Fair comparison (5 models)
All models evaluated on `to_Q`. Model 3 (Player + Team Context) wins 13/17 qualities. Includes case study notebooks and a Streamlit app for interactive exploration.

### v4 — Per-quality tactical models (current)

For each of the 8 midfielder qualities, an exhaustive search tests all 127 combinations of the 7 team tactical dimensions to find the subset that maximises out-of-sample R². Every tactical model is benchmarked against a **naive baseline** that only uses the pre-transfer quality (regression to the mean).

**Model formulation:**  ΔQᵢ = α + β · Qᵢᵖʳᵉ + Σ γₖ · ΔTQₖ  (subset Sᵢ selected per quality)

**Sample:** 4,888 midfielders (train 3,910 / test 978).

| Quality | Baseline R² | Tactical R² | R² gain | Selected team qualities |
|---------|------------|------------|---------|------------------------|
| Involvement | 0.208 | 0.329 | +58.7% | Attack, Atk Trans, Defence, Def Trans, Outcome |
| Effectiveness | 0.251 | 0.292 | +16.2% | Attack, Atk Trans, Defence, Outcome, Penetration |
| Providing teammates | 0.205 | 0.241 | +17.4% | Attack, Defence, Outcome |
| Passing quality | 0.185 | 0.247 | +33.8% | Attack, Defence, Def Trans, Outcome |
| Intelligent defence | 0.234 | 0.249 | +6.6% | Chance Creation, Defence, Def Trans, Outcome |
| Active defence | 0.228 | 0.236 | +3.7% | Defence, Outcome |
| Box threat | 0.194 | 0.222 | +14.5% | Attack, Chance Creation, Defence, Outcome |
| Progression | 0.195 | 0.214 | +9.5% | Attack, Atk Trans, Defence, Def Trans |

**Key findings:** Defence is selected for all 8 qualities; Outcome for 7/8; Attack for 6/8. Involvement benefits most from tactical context (+59% R²). The tactical model helps more when the tactical change is larger — for transfers between similar teams, the gain is near zero. Case studies include Declan Rice, Kovačić, Eriksen, Pjanić, and Romeu. Full results in `summary.ipynb`, segment analysis in `02_segment_analysis.ipynb`.

## Tech stack

Python 3.12 · pandas · statsmodels (OLS) · scikit-learn · plotly · matplotlib · seaborn · Streamlit
