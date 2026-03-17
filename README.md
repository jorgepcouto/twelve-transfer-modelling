# Twelve Transfer Modelling

Master thesis project: predicting how football players' performance qualities change after within-league transfers.

## What this does

Given a player's pre-transfer **Twelve Quality Scores** and the **team tactical styles** of origin and destination clubs, the model predicts post-transfer player qualities. Models are position-specific (currently focused on Midfielders).

## Data sources

- **Twelve Football (Wyscout)** — player performance metrics, team stats, quality scores
- **Transfermarkt** — transfer records, fees, market values

Raw data is stored externally in `../thesis_data/` (not committed).

## Repository structure

```
twelve-transfer-modelling/
├── thesis_model/
│   └── python_notebooks/
│       ├── preprocessing/          # Data pipeline (01–08), produces v5 parquet
│       ├── models/                 # Model v1: 6 separate notebooks
│       ├── thesis_models_v2/       # Model v2: consolidated, regression-to-mean analysis
│       └── thesis_models_v3/       # Model v3 (current): fair comparison, 5 models
├── practice_model/                 # Earlier model iterations (v1, v2)
├── twelve_qualities/               # Quality definitions
│   ├── player_qualities.csv
│   └── team_qualities.txt
└── CLAUDE.md
```

## Data pipeline

The preprocessing pipeline (notebooks 01–08) processes a master dataset (262K transfers, 539 columns) into a modelling-ready parquet:

1. **Master** → 262K rows, 539 cols
2. **Filter** → ~18K rows (within-league, same position, 900+ min both sides)
3. **Clean columns** → remove identity/metadata, drop goalkeepers
4. **Keep core features** → Twelve Quality Scores (20×2) + team qualities (7×2) + structural
5. **Compute team qualities** → 7 tactical style dimensions per team-season

## Models (v3 — current)

Five OLS regression models, all evaluated on `to_Q` (post-transfer quality level):

| # | Model | Features |
|---|-------|----------|
| 1 | Naive Baseline | Same quality pre-transfer (1) |
| 2 | Player Profile | All 17 pre-transfer qualities |
| 3 | Player + Team Context | 17 qualities + 14 team styles (origin + destination) |
| 4 | Tactical Shift | 7 delta team qualities |
| 5 | Player + Tactical Shift | 17 qualities + 7 delta team qualities |

**Key finding**: Model 3 wins 13/17 qualities. No single best model — the best model varies by quality.

## Tech stack

Python 3.12 · pandas · statsmodels (OLS) · scikit-learn · plotly · matplotlib · seaborn
