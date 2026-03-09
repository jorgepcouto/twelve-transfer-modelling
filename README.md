# Twelve Transfer Modelling

Master thesis project: predicting how football players' performance qualities change after within-league transfers.

## What this does

Given a player's pre-transfer **Twelve Quality Scores** and the **team tactical styles** of origin and destination clubs, the model predicts how the player's qualities will change post-transfer. Models are position-specific (Central Defender, Full Back, Midfielder, Winger, Striker).

## Data sources

- **Twelve Football (Wyscout)** — player performance metrics, team stats, quality scores
- **Transfermarkt** — transfer records, fees, market values

Raw data is stored externally in `thesis_data/` (not committed).

## Repository structure

```
twelve-transfer-modelling/
├── thesis_model/
│   └── python_notebooks/       # Active thesis pipeline (run in order)
│       ├── 01_data_exploration.ipynb
│       ├── 02_filter_and_column_review.ipynb
│       ├── 03_column_decisions_and_v2.ipynb
│       ├── 03_1_remove_goalkeepers.ipynb
│       ├── 04_player_performance_deep_dive.ipynb
│       └── 05_drop_redundant_metrics_and_v4.ipynb
├── practice_model/             # Earlier model iterations (v1, v2)
├── twelve_qualities/           # Quality definitions
│   ├── player_qualities.csv    # 20 Twelve Quality Score formulas
│   └── team_qualities.txt      # 7 team style quality definitions
└── CLAUDE.md
```

## Data pipeline

The thesis notebook pipeline processes a master dataset (262K transfers, 539 columns) down to a clean modelling-ready parquet:

1. **Master** → 262K rows, 539 cols (all transfers, all metrics)
2. **Filter** → ~20K rows (within-league, same position, 900+ min both sides)
3. **Drop metadata** → remove 28 identity/redundant columns
4. **Remove goalkeepers** → 5 outfield positions only
5. **Keep core features** → Twelve Quality Scores (20×2) + team stats + structural columns

See `CLAUDE.md` for detailed architecture notes.

## Tech stack

Python 3.12 · pandas · scikit-learn · xgboost · matplotlib · seaborn · plotly
