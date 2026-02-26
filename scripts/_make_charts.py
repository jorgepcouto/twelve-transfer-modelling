import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams['font.family'] = 'Helvetica Neue'
plt.rcParams['font.size'] = 12

base = Path("/Users/jorgepadilla/Documents") / "Documents - Jorge\u2019s MacBook Air" / "thesis_data" / "raw_data"
df = pd.read_parquet(base / "Transfers" / "transfers_model_v2_2018_2025.parquet")

sections = {}
for c in df.columns:
    if c.startswith("tm_"): sections.setdefault("tm_transfer", []).append(c)
    elif c.startswith("from_team_stats_"): sections.setdefault("from_team_stats", []).append(c)
    elif c.startswith("from_comp_"): sections.setdefault("from_comp_meta", []).append(c)
    elif c.startswith("from_"): sections.setdefault("from_player", []).append(c)
    elif c.startswith("to_team_stats_"): sections.setdefault("to_team_stats", []).append(c)
    elif c.startswith("to_comp_"): sections.setdefault("to_comp_meta", []).append(c)
    elif c.startswith("to_"): sections.setdefault("to_player", []).append(c)
    elif c.startswith("wyscout_"): sections.setdefault("wyscout_player", []).append(c)
    else: sections.setdefault("player_meta", []).append(c)

labels = {
    "player_meta": "Player Meta",
    "tm_transfer": "Transfermarkt",
    "wyscout_player": "Wyscout Player",
    "from_player": "Player Stats",
    "from_comp_meta": "Comp Meta",
    "from_team_stats": "Team Stats",
    "to_player": "Player Stats",
    "to_comp_meta": "Comp Meta",
    "to_team_stats": "Team Stats",
}

sample_cols = {
    "player_meta": "wy_player_id, tm_player_id,\nshort_name, birth_date,\ntransfer_type ...",
    "tm_transfer": "transfer_date, transfer_value,\ntransfer_fee, remaining_contract,\ncontract_until_date",
    "wyscout_player": "first_name, last_name, height,\nweight, foot, role,\npassport, birth_country ...",
    "from_player": "5 meta + 50 raw + 46 per90\n+ 75 z-scores",
    "from_comp_meta": "name, country, division,\nstart_date, end_date ...",
    "from_team_stats": "74 team season metrics:\nxg, goals, ppda, field_tilt,\npossession, recoveries ...",
    "to_player": "5 meta + 50 raw + 46 per90\n+ 75 z-scores",
    "to_comp_meta": "name, country, division,\nstart_date, end_date ...",
    "to_team_stats": "74 team season metrics:\nxg, goals, ppda, field_tilt,\npossession, recoveries ...",
}

# Blue-gray palette
lc = {
    "player_meta": "#1565C0",
    "tm_transfer": "#1976D2",
    "wyscout_player": "#1E88E5",
    "from_player": "#546E7A",
    "from_comp_meta": "#607D8B",
    "from_team_stats": "#78909C",
    "to_player": "#546E7A",
    "to_comp_meta": "#607D8B",
    "to_team_stats": "#78909C",
}

# =========================================================
# GRAPH 1: TREE
# =========================================================
fig, ax = plt.subplots(figsize=(26, 19), facecolor="white")
ax.set_xlim(0, 26)
ax.set_ylim(0, 19)
ax.axis("off")

ax.text(13, 18.3, "Transfer Dataset Structure", fontsize=32, fontweight="bold",
        ha="center", va="center", color="#212121")
ax.text(13, 17.5, f"{len(df):,} rows   |   {len(df.columns)} columns   |   Seasons 2018\u20132025",
        fontsize=17, ha="center", va="center", color="#757575")

# Root
rx, ry = 13, 16.0
ax.add_patch(mpatches.FancyBboxPatch((rx - 2.75, ry - 0.5), 5.5, 1.0,
    boxstyle="round,pad=0.15", facecolor="#1A237E", edgecolor="none", zorder=3))
ax.text(rx, ry, "TRANSFER RECORD", fontsize=18, fontweight="bold",
    ha="center", va="center", color="white", zorder=4)

# Branches
bdata = [
    ("GLOBAL", 4.5, "#37474F", ["player_meta", "tm_transfer", "wyscout_player"]),
    ("FROM  (origin club)", 13.0, "#455A64", ["from_player", "from_comp_meta", "from_team_stats"]),
    ("TO  (destination club)", 21.5, "#546E7A", ["to_player", "to_comp_meta", "to_team_stats"]),
]

by = 13.7
lsy = 11.5
lsp = 3.8

for bl, bx, bc, lsecs in bdata:
    ax.plot([rx, bx], [ry - 0.5, by + 0.45], color="#B0BEC5", lw=2.5, zorder=1)
    ax.add_patch(mpatches.FancyBboxPatch((bx - 2.8, by - 0.4), 5.6, 0.8,
        boxstyle="round,pad=0.12", facecolor=bc, edgecolor="none", zorder=3))
    ax.text(bx, by, bl, fontsize=15, fontweight="bold",
        ha="center", va="center", color="white", zorder=4)

    for i, s in enumerate(lsecs):
        ly = lsy - i * lsp
        ax.plot([bx, bx], [by - 0.4, ly + 1.2], color="#CFD8DC", lw=1.8, zorder=1)
        ax.add_patch(mpatches.FancyBboxPatch((bx - 3.5, ly - 1.15), 7.0, 2.3,
            boxstyle="round,pad=0.15", facecolor=lc[s], edgecolor="none", alpha=0.92, zorder=3))
        n = len(sections[s])
        ax.text(bx, ly + 0.65, f"{labels[s]}  ({n} cols)", fontsize=15, fontweight="bold",
            ha="center", va="center", color="white", zorder=4)
        ax.plot([bx - 3.0, bx + 3.0], [ly + 0.25, ly + 0.25],
                color="white", lw=0.7, alpha=0.35, zorder=4)
        ax.text(bx, ly - 0.35, sample_cols[s], fontsize=11.5,
            ha="center", va="center", color="white", alpha=0.88, zorder=4, linespacing=1.35)

# Footnotes
fn = [
    "* Transfermarkt coverage (26.6%) limited to ~34.5K of 83.5K unique players with a TM transfer history.",
    "* Wyscout Player meta (22.5%) only covers players present in the Wyscout database at extraction time.",
    "* Comp Meta (~73%) missing for competitions not in the Wyscout competitions catalog (lower leagues, friendlies).",
    "* Team Stats (~91%) missing when (team, competition, season) not found in team_stats_season source.",
    "* tm_transfer_fee > 0 in only 2.4% of rows \u2014 most transfers are free, loans, or undisclosed.",
]
ax.text(1.2, 2.5, "Notes", fontsize=15, fontweight="bold", color="#424242", va="top")
for i, note in enumerate(fn):
    ax.text(1.2, 1.8 - i * 0.55, note, fontsize=11.5, color="#616161", va="top")

plt.savefig(str(base.parent / "dataset_tree_v2.png"), dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: dataset_tree_v2.png")
plt.close()

# =========================================================
# GRAPH 2: COVERAGE
# =========================================================
fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor="white")

order = ["player_meta", "tm_transfer", "wyscout_player",
         "from_player", "from_comp_meta", "from_team_stats",
         "to_player", "to_comp_meta", "to_team_stats"]

covs = []
for s in order:
    if s == "player_meta":
        covs.append(100.0)
    elif s == "tm_transfer":
        covs.append(df["tm_transfer_date"].notna().sum() / len(df) * 100)
    else:
        covs.append(df[sections[s][0]].notna().sum() / len(df) * 100)

bl2 = [f"{labels[s]}  ({len(sections[s])})" for s in order]
yp = np.arange(len(order))[::-1]
bars = ax2.barh(yp, covs, color=[lc[s] for s in order], height=0.55, edgecolor="white", linewidth=1)

for b, cv in zip(bars, covs):
    ax2.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2,
             f"{cv:.1f}%", va="center", fontsize=14, fontweight="bold", color="#333")

for x in [25, 50, 75, 100]:
    ax2.axvline(x, color="#ECEFF1", lw=0.8, zorder=0)

ax2.set_yticks(yp)
ax2.set_yticklabels(bl2, fontsize=14)
ax2.set_xlim(0, 115)
ax2.set_xlabel("Coverage  (% of 262,340 rows with data)", fontsize=14)
ax2.set_title("Dataset Coverage by Section", fontsize=22, fontweight="bold", pad=20, color="#212121")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color("#B0BEC5")
ax2.spines["left"].set_color("#B0BEC5")
ax2.tick_params(colors="#555")
ax2.text(0.5, 1.04,
         f"transfers_model_v2_2018_2025.parquet   |   {len(df):,} rows   |   {len(df.columns)} columns",
         transform=ax2.transAxes, fontsize=13, ha="center", color="#90A4AE")

plt.tight_layout()
plt.savefig(str(base.parent / "dataset_coverage_v2.png"), dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: dataset_coverage_v2.png")
plt.close()
