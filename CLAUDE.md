# CLAUDE.md

## Path setup — DO THIS FIRST

The repo path has spaces and an apostrophe. **Always use relative paths from the working directory** (`twelve-transfer-modelling/` or its parent `thesis/`). Never `cd` with the absolute path — it will break. If you must use absolute paths, use `find` or `python3 os.path`:

```bash
# From the thesis/ working directory, everything works with relative paths:
ls twelve-transfer-modelling/thesis_model/thesis_models_v4/
cat twelve-transfer-modelling/thesis_model/thesis_models_v4/summary.ipynb

# For reading notebooks, always use python3:
python3 -c "
import json
with open('twelve-transfer-modelling/thesis_model/thesis_models_v4/summary.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f'=== Cell {i} ({cell[\"cell_type\"]}) ===')
    print(''.join(cell['source'])[:500])
"
```

## Project overview

Master thesis: **predicting how football players' performance qualities change after within-league transfers**, using Twelve Quality Scores and team tactical styles. OLS regression: given a player's pre-transfer qualities and tactical style changes (origin → destination team), predict post-transfer qualities.

## Data

Raw data lives in `../thesis_data/` (never committed):
- `raw_data/Wyscout/` — player metrics, team stats (`wyscout_teams.parquet`)
- `raw_data/Teams_stats/`, `raw_data/Transfers/`
- `processed_data/thesis_model_dataset/active/within_league_transfers_v5.parquet` — **the modelling dataset**

**Dataset stats**: 262K transfers → filtered to ~18K (within-league, same position, 900+ min) → v5 parquet.

## Repository structure

```
twelve-transfer-modelling/
├── thesis_model/
│   ├── preprocessing/           # 01–08: raw → v5 parquet
│   ├── thesis_models_v1/        # 6 OLS notebooks (midfielders, deprecated)
│   ├── thesis_models_v2/        # Consolidated, regression-to-mean discovery
│   ├── thesis_models_v3/        # 5-model comparison, case studies, Streamlit app
│   │   └── streamlit_app/
│   └── thesis_models_v4/        # ← CURRENT: per-quality tactical models
│       ├── {quality}_model/     # One folder per quality (8 total)
│       │   └── 01_{quality}_search.ipynb
│       └── summary.ipynb        # Cross-quality comparison & case studies
├── practice_model/              # Old exploratory work (ignore)
├── twelve_qualities/
│   ├── player_qualities.csv     # 20 quality definitions with weights
│   ├── team_qualities.txt       # 7 team style definitions (Python enum)
│   └── team.py
└── CLAUDE.md
```

## Current work: v4 — Per-quality tactical models

### Approach
For each of 8 midfielder qualities, exhaustive search across all 127 combinations of 7 team tactical dimensions. Select the subset that maximises **out-of-sample R²**. Benchmark against a naive baseline (pre-quality only = regression to the mean).

### Formulation
```
ΔQᵢ = α + β · Qᵢᵖʳᵉ + Σ γₖ · ΔTQₖ   (subset Sᵢ per quality)
```

### Sample
4,888 midfielders — train: 3,910, test: 978.

### Results

| Quality             | Baseline R² | Tactical R² | Gain    | Selected team qualities                              |
|---------------------|-------------|-------------|---------|------------------------------------------------------|
| Involvement         | 0.208       | 0.329       | +58.7%  | Attack, Atk Trans, Defence, Def Trans, Outcome       |
| Passing quality     | 0.185       | 0.247       | +33.8%  | Attack, Defence, Def Trans, Outcome                  |
| Providing teammates | 0.205       | 0.241       | +17.4%  | Attack, Defence, Outcome                             |
| Effectiveness       | 0.251       | 0.292       | +16.2%  | Attack, Atk Trans, Defence, Outcome, Penetration     |
| Box threat          | 0.194       | 0.222       | +14.5%  | Attack, Chance Creation, Defence, Outcome            |
| Progression         | 0.195       | 0.214       | +9.5%   | Attack, Atk Trans, Defence, Def Trans                |
| Intelligent defence | 0.234       | 0.249       | +6.6%   | Chance Creation, Defence, Def Trans, Outcome         |
| Active defence      | 0.228       | 0.236       | +3.7%   | Defence, Outcome                                     |

### Key findings
- **Defence** selected for all 8 qualities; **Outcome** for 7/8; **Attack** for 6/8
- Involvement benefits most from tactical context (+59% R²)
- Defensive qualities (Active/Intelligent defence) gain least — mostly regression to the mean
- Case studies: Declan Rice, Kovačić, Eriksen, Pjanić, Romeu, Saúl, Çalhanoğlu

## Key domain concepts

**Player qualities** (20 total, 17 for midfielders): z-scored performance dimensions. Each transfer has `from_Q` (pre) and `to_Q` (post). The 8 modelled in v4: Involvement, Active defence, Intelligent defence, Progression, Passing quality, Effectiveness, Providing teammates, Box threat.

**Team tactical qualities** (7): Defence, Def Transition, Atk Transition, Attack, Penetration, Chance Creation, Outcome. Z-scored per team-season. Delta = destination − origin.

**Positions**: Central Defender, Full Back, Midfielder, Winger, Striker. Current models: **Midfielders only**.

## Notebook conventions

- Keep notebooks concise — no walls of stats
- Show **top-5 results per model**, not mean R²
- Use the design system from summary.ipynb (BG='#FAFAFA', C_BASELINE='#BF5B3F', C_TACTICAL='#2E74B5', C_POST='#1A9C6E')
- Read notebooks with `python3 + json` (not `cat`)

## Tech stack

Python 3.12 · pandas · statsmodels (OLS) · scikit-learn · matplotlib · seaborn · Streamlit
