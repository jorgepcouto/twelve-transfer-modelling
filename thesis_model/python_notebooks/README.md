# Thesis Model — Notebooks

## Structure

```
python_notebooks/
├── preprocessing/           ← Data pipeline (01–08, already executed)
├── models/                  ← Model v1: 6 separate notebooks per model
├── thesis_models_v2/        ← Model v2: consolidated, regression-to-mean analysis
├── thesis_models_v3/        ← Model v3 (current): fair comparison, 5 models
└── README.md
```

## Preprocessing pipeline (complete)

| # | Notebook | What it does | Output |
|---|----------|-------------|--------|
| 01 | data_exploration | EDA on master parquet (262K rows, 539 cols) | — |
| 02 | filter_and_column_review | Filter to within-league transfers (~18K rows) | `within_league_transfers.parquet` |
| 03 | column_decisions_and_v2 | Drop 28 identity/metadata columns | `within_league_transfers_v2.parquet` |
| 03.1 | remove_goalkeepers | Remove goalkeeper rows | `within_league_transfers_v3.parquet` |
| 04 | player_performance_deep_dive | Per-position analysis of all metric groups | — |
| 05 | drop_redundant_metrics_and_v4 | Keep only Twelve QS + team stats + structural | `within_league_transfers_v4.parquet` |
| 06 | team_qualities | Compute 7 team style qualities from z-scores | `team_qualities.parquet` |
| 07 | join_team_qualities | Join team qualities to transfers | `within_league_transfers_v5.parquet` |
| 08 | v5_final_review | Pre-modelling audit of v5 | — |
| 08.1 | comp_metadata_investigation | Investigate competitions missing metadata | `competitions_missing_metadata.csv` |

## Models

### v1 (`models/`)

6 separate notebooks, one per model. Midfielders only. OLS via statsmodels.

| Notebook | Model |
|----------|-------|
| 01_naive_baseline_midfielders | `to_Q_i = α + β·from_Q_i` |
| 02_all_pre_qualities_midfielders | `to_Q_i = α + Σβ_j·from_Q_j` |
| 03_pre_qualities_plus_team_midfielders | `to_Q_i = α + Σβ_j·from_Q_j + Σγ_k·team_from_k + Σδ_k·team_to_k` |
| 04a_delta_team_to_delta_player_midfielders | `ΔPQ_i = α + Σγ_k·ΔTQ_k` |
| 04b_pre_plus_delta_team_midfielders | `ΔPQ_i = α + Σβ_j·from_Q_j + Σγ_k·ΔTQ_k` |
| 04c_plus_controls_midfielders | `ΔPQ_i = α + Σβ_j·from_Q_j + Σγ_k·ΔTQ_k + δ·age + ε·Δmin` |

### v2 (`thesis_models_v2/`)

Consolidated into 2 notebooks. Discovered that delta models (4b/4c) have inflated R² due to regression-to-mean — the `from_Q` coefficient ≈ -1 means the model mostly learns that extreme values regress to the center.

### v3 (`thesis_models_v3/`) — current

2 notebooks, 5 models. Fair comparison: all models evaluated on `to_Q` (delta models reconstruct `to_Q = from_Q + predicted_delta`). Dropped controls model (age + minutes added marginal gains).

| # | Model | Key insight |
|---|-------|-------------|
| 1 | Naive Baseline | R² ~0.01–0.47 depending on quality |
| 2 | Player Profile | Cross-quality effects improve all 17 qualities |
| 3 | Player + Team Context | **Wins 13/17 qualities**. Best overall. |
| 4 | Tactical Shift | Team change alone explains almost nothing (R² < 0.04) |
| 5 | Player + Tactical Shift | Wins 4/17 by small margins over Model 3 |

Working dataset: `thesis_model_dataset/active/within_league_transfers_v5.parquet`
