# Transfer Modelling: Team Tactical DNA & Player Archetypes

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Analysis of football playing styles and player role classification using match-level data from **Twelve Football (Wyscout)** and transfer data from **Transfermarkt**. Built as part of a master's thesis on data-driven player transfer valuation.

---

## Project Structure

```
twelve-transfer-modelling/          # This repo (code)
├── clustering/
│   ├── team_tactical_dna.ipynb     # Team playing style families (k=5)
│   └── player_archetypes.ipynb     # Player sub-role classification
├── data_processing/                # Notebooks that built the master dataset
│   ├── explore_transfers.ipynb
│   ├── column_alignment_and_reorder.ipynb
│   ├── wyscout_metadata_merge.ipynb
│   ├── team_stats_merge.ipynb
│   └── final_validation.ipynb
├── first_data_review/              # Initial data exploration
├── archive/                        # Earlier clustering iterations
├── scripts/                        # Utility scripts
├── streamlit_app/                  # Interactive dashboard
└── README.md

thesis_data/                        # Sibling folder (data, not in git)
├── raw_data/
│   ├── Teams_stats/
│   │   └── team_stats_season.parquet
│   ├── Transfers/
│   │   ├── male_transfer_model.parquet
│   │   └── transfers_same_competitions_data.parquet
│   ├── Wyscout/
│   │   ├── competitions_wyscout.parquet
│   │   └── players_wyscout.parquet
│   └── Transfermarkt/
│       ├── tm_teams.parquet
│       └── transfer_history_all.parquet
└── processed_data/
    ├── master_dataset/
    │   └── transfers_model_v2_2018_2025.parquet
    ├── team_styles/
    │   ├── team_qualities.parquet        (output of team notebook)
    │   └── team_style_clusters.parquet   (output of team notebook)
    └── player_archetypes/
        └── player_archetypes.parquet     (output of player notebook)
```

---

## Data Setup

The notebooks expect a **sibling directory** called `thesis_data` next to `twelve-transfer-modelling`. Both folders must live in the same parent directory.

### Required files to run the clustering notebooks

Download or copy these files into the exact paths shown below:

| # | File | Path inside `thesis_data/` | Source | Size |
|---|------|---------------------------|--------|------|
| 1 | `competitions_wyscout.parquet` | `raw_data/Wyscout/` | Wyscout metadata export | 39 KB |
| 2 | `team_stats_season.parquet` | `raw_data/Teams_stats/` | Twelve Football API | 12 MB |
| 3 | `transfers_model_v2_2018_2025.parquet` | `processed_data/master_dataset/` | Built via `data_processing/` notebooks | 468 MB |

> **Files 1-2** are raw API exports. **File 3** is the unified transfer dataset built by merging Twelve Football transfer data with Transfermarkt valuations and Wyscout metadata. If you only have the raw data, run the `data_processing/` notebooks first (see below).

### Quick setup (copy-paste)

```bash
# From the parent directory containing both folders:
mkdir -p thesis_data/raw_data/Teams_stats
mkdir -p thesis_data/raw_data/Wyscout
mkdir -p thesis_data/processed_data/master_dataset
mkdir -p thesis_data/processed_data/team_styles
mkdir -p thesis_data/processed_data/player_archetypes

# Then copy your parquet files into the directories above
```

---

## Running the Notebooks

### Dependencies

```bash
pip install pandas numpy matplotlib plotly scikit-learn umap-learn
```

### Option A: Run only the analysis (recommended)

If you already have `transfers_model_v2_2018_2025.parquet`:

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `clustering/team_tactical_dna.ipynb` | Computes 7 tactical qualities per team-season, clusters into 5 playing style families, visualizes EPL trends |
| 2 | `clustering/player_archetypes.ipynb` | Classifies players into position-specific archetypes using K-Means and GMM, shows famous player role evolution |

Both notebooks are **self-contained** -- each one loads data, processes it, and produces all visualizations in a single run. They also export their results to `processed_data/`.

### Option B: Build everything from raw data

If you only have the raw API exports, run the data processing pipeline first:

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `data_processing/explore_transfers.ipynb` | Validates the two raw transfer parquets are complementary |
| 2 | `data_processing/column_alignment_and_reorder.ipynb` | Aligns column names across datasets |
| 3 | `data_processing/wyscout_metadata_merge.ipynb` | Merges Wyscout player/competition metadata |
| 4 | `data_processing/team_stats_merge.ipynb` | Merges team stats with transfer data |
| 5 | `data_processing/final_validation.ipynb` | Final QA checks on the unified dataset |
| 6 | `clustering/team_tactical_dna.ipynb` | Team style analysis |
| 7 | `clustering/player_archetypes.ipynb` | Player archetype classification |

### Additional raw data for full pipeline

| File | Path inside `thesis_data/` | Source |
|------|---------------------------|--------|
| `male_transfer_model.parquet` | `raw_data/Transfers/` | Twelve Football API (cross-competition transfers) |
| `transfers_same_competitions_data.parquet` | `raw_data/Transfers/` | Twelve Football API (same-competition transfers) |
| `tm_teams.parquet` | `raw_data/Transfermarkt/` | Transfermarkt scrape |
| `transfer_history_all.parquet` | `raw_data/Transfermarkt/` | Transfermarkt scrape |

---

## What the Analysis Produces

### Team Tactical DNA

Every team-season gets scored on **7 tactical dimensions** computed as z-scores within each (competition, season):

| Dimension | What it measures (low end vs high end) |
|-----------|---------------------------------------|
| Defence | Low Block vs High Press |
| Def. Transition | Regroup vs Counter-Press |
| Att. Transition | Build-Up vs Counter-Attack |
| Attack | Short Passing vs Direct/Long |
| Penetration | Crosses vs Carries |
| Chance Creation | Sustained vs Direct Chances |
| Outcome | Expected + actual points |

Teams are then grouped into **5 tactical families** based on radar chart similarity.

### Player Archetypes

Players are classified into sub-roles within each position:

| Position | # Archetypes | Method |
|----------|-------------|--------|
| Central Defender | 4 | K-Means + GMM |
| Full Back | 3 | K-Means + GMM |
| Midfielder | 5 | K-Means + GMM |
| Winger | 5 | K-Means + GMM |
| Striker | 4 | K-Means + GMM |

GMM provides **soft probabilities** (e.g., "91% Finisher, 5% Target Man, 4% Runner"), enabling role evolution tracking across seasons.

---

## Key Assumptions

1. All tactical scores are **z-scored within (competition, season)** -- a La Liga team is compared to other La Liga teams that same year, not to the Allsvenskan.
2. Analysis is restricted to **Division 1 (top-flight)** leagues only.
3. Goalkeepers are **excluded** from player archetype clustering.
4. Each dimension is a **style scale**, not a quality rating -- scoring high on "Sustained vs Direct Chances" means the team creates more directly, not that they create more overall.
5. The `higher_is_better` flags in the QUALITIES dictionary determine the polarity of each metric within its dimension.

---

## Tech Stack

- **pandas** + **numpy** for data processing
- **plotly** for interactive premium visualizations
- **scikit-learn** for K-Means clustering and preprocessing
- **scikit-learn (GaussianMixture)** for soft probability clustering
- **umap-learn** for dimensionality reduction
- **matplotlib** for supplementary static charts
