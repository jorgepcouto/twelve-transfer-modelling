"""
Cross-Division Transfer Performance Comparison
================================================
Streamlit app to visualize player performance before and after
transfers between divisions within the same country.

Countries: Mexico, Sweden, England
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(
    page_title="Cross-Division Transfer Analysis",
    page_icon="⚽",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────

COMPETITION_CONFIG = {
    "Mexico 🇲🇽": {
        "leagues": {615: "Liga de Expansión MX", 617: "Liga MX"},
        "pairs": [(615, 617), (617, 615)],
    },
    "Sweden 🇸🇪": {
        "leagues": {808: "Allsvenskan", 818: "Superettan"},
        "pairs": [(808, 818), (818, 808)],
    },
    "England 🏴󠁧󠁢󠁥󠁮󠁧󠁿": {
        "leagues": {346: "Championship", 364: "Premier League"},
        "pairs": [(346, 364), (364, 346)],
    },
}

RADAR_METRICS = [
    "Hold-up play", "Involvement", "Providing teammates", "Aerial threat",
    "Poaching", "Run quality", "Pressing", "Finishing", "Box threat",
    "Dribbling", "Active defence", "Progression", "Intelligent defence",
    "Defensive heading", "Passing quality", "Winning duels", "Composure",
    "Effectiveness", "Territorial dominance", "Chance prevention",
]

ATTACKING_METRICS = [
    "xG per 90", "Goals per 90", "xGOT per 90", "Shot conversion %",
    "Goals per box touch", "xG per box touch", "xG per shot",
    "Touches in box per 90", "Box entries per 90",
    "Dribbles (xT) per 90", "Dribbles success %",
    "Goals - xG", "xGDribble per 90",
]

DEFENSIVE_METRICS = [
    "Defensive actions per 90", "True tackles won per 90", "Tackle success %",
    "Interceptions per 90", "Defending 1v1 %",
    "Defensive duels won per 90", "Defensive duels won %",
    "Defensive aerials won per 90", "Defensive aerials won %",
    "Ball recoveries per 90", "High recoveries per 90",
    "Possessions won per 90", "Counterpressing per 90",
]

PASSING_METRICS = [
    "xA per 90", "Assists per 90", "Key passes per 90",
    "Creative passes per 90", "Playmaking passes per 90",
    "xGCreated per 90", "xGBuildup per 90",
    "Passes (xT) per 90", "Deep completions per 90",
    "Ball progression (xT) per 90", "Carries (xT) per 90",
    "Ball runs (xT) per 90", "Deep runs (xT) per 90",
    "Crosses (xT) per 90",
]

PHYSICAL_METRICS = [
    "Aerials per 90", "Aerials won %", "Aerials won per 90",
    "Headed plays per 90", "Pressure resistance %",
    "Touches per 90", "Linkups per 90", "Losses per 90",
    "High turnovers per 90", "xGChain per possession",
    "xG + xA per 100 touches",
]


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load transfer data, competition metadata, and team name lookup."""
    base = Path(__file__).parent

    df = pd.read_parquet(
        base / "../../thesis_data/raw_data_twelve/Twelve/male_transfer_model.parquet"
    )
    comps = pd.read_parquet(
        base / "../../thesis_data/raw_data_twelve/Wyscout/competitions_wyscout.parquet"
    )

    # Team name lookup from processed data
    team_names = {}
    clean_path = base / "../../thesis_data/processed/transfers_clean.parquet"
    if clean_path.exists():
        clean = pd.read_parquet(
            clean_path,
            columns=["from_team_id", "to_team_id", "tm_team_from", "tm_team_to"],
        )
        for _, row in clean.dropna(subset=["tm_team_from"]).iterrows():
            team_names[int(row["from_team_id"])] = row["tm_team_from"]
        for _, row in clean.dropna(subset=["tm_team_to"]).iterrows():
            team_names[int(row["to_team_id"])] = row["tm_team_to"]

    return df, comps, team_names


def filter_cross_division(df, country):
    """Filter to cross-division transfers for a country, same position only."""
    config = COMPETITION_CONFIG[country]
    comp_ids = list(config["leagues"].keys())

    mask = (
        df["from_competition"].isin(comp_ids)
        & df["to_competition"].isin(comp_ids)
        & (df["from_competition"] != df["to_competition"])
        & (df["from_position"] == df["to_position"])
    )
    return df[mask].copy()


def team_name(team_id, team_names):
    """Return team name or fallback to ID."""
    return team_names.get(int(team_id), f"Team {team_id}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df, team_names):
    """Render cascading filters. Returns (country, filtered_df, player_row)."""
    st.sidebar.header("⚽ Filters")

    # 1. Country
    country = st.sidebar.selectbox("Country", list(COMPETITION_CONFIG.keys()))
    filtered = filter_cross_division(df, country)
    config = COMPETITION_CONFIG[country]
    league_names = config["leagues"]

    # 2. Direction
    directions = (
        filtered.groupby(["from_competition", "to_competition"])
        .size()
        .reset_index(name="n")
    )
    dir_options = []
    for _, row in directions.iterrows():
        fr = league_names[row["from_competition"]]
        to = league_names[row["to_competition"]]
        dir_options.append((row["from_competition"], row["to_competition"], f"{fr} → {to}"))

    if not dir_options:
        st.sidebar.warning("No transfers found for this country.")
        return country, filtered, pd.DataFrame()

    sel_dir = st.sidebar.selectbox(
        "Transfer Direction", dir_options, format_func=lambda x: x[2]
    )
    filtered = filtered[
        (filtered["from_competition"] == sel_dir[0])
        & (filtered["to_competition"] == sel_dir[1])
    ]

    # 3. Include promotion/relegation
    include_promo = st.sidebar.checkbox(
        "Include promotion/relegation (same team)", value=True
    )
    if not include_promo:
        filtered = filtered[filtered["from_team_id"] != filtered["to_team_id"]]

    if filtered.empty:
        st.sidebar.warning("No transfers match these filters.")
        return country, filtered, pd.DataFrame()

    # 4. Season
    seasons = sorted(filtered["to_season"].unique())
    sel_season = st.sidebar.selectbox(
        "Destination Season", ["All"] + [str(s) for s in seasons]
    )
    if sel_season != "All":
        filtered = filtered[filtered["to_season"] == int(sel_season)]

    # 5. Team
    team_ids = sorted(filtered["to_team_id"].unique())
    team_opts = {tid: team_name(tid, team_names) for tid in team_ids}
    sel_team = st.sidebar.selectbox(
        "Destination Team",
        ["All"] + team_ids,
        format_func=lambda x: "All" if x == "All" else team_opts.get(x, str(x)),
    )
    if sel_team != "All":
        filtered = filtered[filtered["to_team_id"] == sel_team]

    if filtered.empty:
        st.sidebar.warning("No transfers match these filters.")
        return country, filtered, pd.DataFrame()

    # 6. Player
    players = (
        filtered[["player_id", "short_name"]]
        .drop_duplicates()
        .sort_values("short_name")
    )
    player_opts = dict(zip(players["player_id"], players["short_name"]))
    sel_player = st.sidebar.selectbox(
        "Player",
        list(player_opts.keys()),
        format_func=lambda x: player_opts[x],
    )

    player_data = filtered[filtered["player_id"] == sel_player]

    # If multiple transfers for same player, pick one
    if len(player_data) > 1:
        labels = [
            f"{int(r['from_season'])} → {int(r['to_season'])}"
            for _, r in player_data.iterrows()
        ]
        idx = st.sidebar.selectbox("Transfer window", range(len(labels)), format_func=lambda i: labels[i])
        player_data = player_data.iloc[[idx]]

    st.sidebar.divider()
    st.sidebar.caption(f"📊 {len(filtered)} records | {filtered['player_id'].nunique()} players")

    return country, filtered, player_data


# ── Player Header ────────────────────────────────────────────────────────────

def render_header(player_row, country, team_names):
    """Show player info banner."""
    r = player_row.iloc[0]
    config = COMPETITION_CONFIG[country]
    ln = config["leagues"]

    from_league = ln.get(int(r["from_competition"]), str(r["from_competition"]))
    to_league = ln.get(int(r["to_competition"]), str(r["to_competition"]))
    from_team = team_name(r["from_team_id"], team_names)
    to_team = team_name(r["to_team_id"], team_names)
    is_same_team = int(r["from_team_id"]) == int(r["to_team_id"])
    transfer_type = "🔄 Promotion/Relegation" if is_same_team else "➡️ Transfer"

    st.markdown(f"## {r['short_name']}  &nbsp; `{r['from_position']}`")
    st.markdown(
        f"**{from_league}** ({from_team}) → **{to_league}** ({to_team}) "
        f"&nbsp; | &nbsp; {transfer_type}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("From Season", int(r["from_season"]))
    c2.metric("To Season", int(r["to_season"]))
    c3.metric("Minutes Before", f"{r['from_Minutes']:.0f}")
    c4.metric(
        "Minutes After",
        f"{r['to_Minutes']:.0f}" if pd.notna(r.get("to_Minutes")) else "N/A",
    )


# ── Radar Chart ──────────────────────────────────────────────────────────────

def render_radar(player_row):
    """Plotly radar chart for the 20 composite metrics."""
    r = player_row.iloc[0]
    labels, from_vals, to_vals = [], [], []

    for m in RADAR_METRICS:
        fv = r.get(f"from_{m}")
        tv = r.get(f"to_{m}")
        if pd.notna(fv) and pd.notna(tv):
            labels.append(m)
            from_vals.append(float(fv))
            to_vals.append(float(tv))

    if not labels:
        st.warning("No composite metrics available for this player.")
        return

    # Close the polygon
    labels_closed = labels + [labels[0]]
    from_closed = from_vals + [from_vals[0]]
    to_closed = to_vals + [to_vals[0]]

    all_vals = from_vals + to_vals
    rng = max(abs(min(all_vals)), abs(max(all_vals))) * 1.15

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=from_closed, theta=labels_closed, fill="toself",
        name="Before", opacity=0.5, line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=to_closed, theta=labels_closed, fill="toself",
        name="After", opacity=0.5, line=dict(color="#EF553B", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-rng, rng])),
        showlegend=True,
        height=600,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table below radar
    radar_df = pd.DataFrame({
        "Metric": labels,
        "Before": from_vals,
        "After": to_vals,
        "Delta": [t - f for f, t in zip(from_vals, to_vals)],
    }).sort_values("Delta", ascending=False)
    radar_df[["Before", "After", "Delta"]] = radar_df[["Before", "After", "Delta"]].round(2)
    st.dataframe(radar_df, use_container_width=True, hide_index=True)


# ── Metric Comparison (Tabs 2-5) ────────────────────────────────────────────

def render_metric_comparison(player_row, metrics, tab_title, use_zscore=False):
    """Bar chart + delta indicators for a metric group."""
    r = player_row.iloc[0]
    data = []

    for m in metrics:
        col_name = f"z_score_{m}" if use_zscore else m
        from_col = f"from_{col_name}"
        to_col = f"to_{col_name}"

        fv = r.get(from_col)
        tv = r.get(to_col)
        if pd.notna(fv) and pd.notna(tv):
            delta = float(tv) - float(fv)
            data.append({
                "metric": m,
                "Before": round(float(fv), 3),
                "After": round(float(tv), 3),
                "Delta": round(delta, 3),
            })

    if not data:
        st.info("No data available for these metrics with this player.")
        return

    mdf = pd.DataFrame(data)

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before", x=mdf["metric"], y=mdf["Before"], marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        name="After", x=mdf["metric"], y=mdf["After"], marker_color="#EF553B",
    ))
    fig.update_layout(
        barmode="group",
        title=f"{tab_title} {'(Z-Scores)' if use_zscore else '(Per 90 / Rate)'}",
        xaxis_tickangle=-45,
        height=480,
        margin=dict(b=160),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta metrics grid
    n_cols = min(len(data), 5)
    for row_start in range(0, len(data), n_cols):
        cols = st.columns(n_cols)
        for i, d in enumerate(data[row_start : row_start + n_cols]):
            cols[i].metric(
                label=d["metric"][:30],
                value=f"{d['After']:.2f}",
                delta=f"{d['Delta']:+.3f}",
            )


# ── LLM Analysis ────────────────────────────────────────────────────────────

def build_llm_prompt(player_row, country, team_names):
    """Build structured prompt for Claude interpretation."""
    r = player_row.iloc[0]
    config = COMPETITION_CONFIG[country]
    ln = config["leagues"]

    from_league = ln.get(int(r["from_competition"]), "?")
    to_league = ln.get(int(r["to_competition"]), "?")
    from_team = team_name(r["from_team_id"], team_names)
    to_team = team_name(r["to_team_id"], team_names)

    # Radar changes
    radar_lines = []
    for m in RADAR_METRICS:
        fv, tv = r.get(f"from_{m}"), r.get(f"to_{m}")
        if pd.notna(fv) and pd.notna(tv):
            radar_lines.append(f"  {m}: {fv:.2f} → {tv:.2f} (Δ {tv - fv:+.2f})")

    # Per-90 changes: collect all, sort, show top movers
    all_per90 = []
    for group in [ATTACKING_METRICS, DEFENSIVE_METRICS, PASSING_METRICS, PHYSICAL_METRICS]:
        for m in group:
            fv, tv = r.get(f"from_{m}"), r.get(f"to_{m}")
            if pd.notna(fv) and pd.notna(tv) and fv != 0:
                pct = (tv - fv) / abs(fv) * 100
                all_per90.append((m, float(fv), float(tv), float(tv - fv), pct))

    all_per90.sort(key=lambda x: x[4], reverse=True)
    top_up = all_per90[:8]
    top_down = all_per90[-8:]

    prompt = f"""You are a football performance analyst. Analyze this player's performance
change after a cross-division transfer. Answer in Spanish.

PLAYER: {r['short_name']}
POSITION: {r['from_position']}
TRANSFER: {from_league} ({from_team}) → {to_league} ({to_team})
SEASONS: {r['from_season']} → {r['to_season']}
MINUTES: {r['from_Minutes']:.0f} → {r.get('to_Minutes', 'N/A')}

COMPOSITE METRICS (standardized, higher = better):
{chr(10).join(radar_lines)}

TOP IMPROVEMENTS (per 90):
{chr(10).join(f'  {m[0]}: {m[1]:.2f} → {m[2]:.2f} ({m[4]:+.1f}%)' for m in top_up)}

TOP DECLINES (per 90):
{chr(10).join(f'  {m[0]}: {m[1]:.2f} → {m[2]:.2f} ({m[4]:+.1f}%)' for m in top_down)}

Provide:
1. Executive summary (2-3 sentences) of performance change.
2. Key strengths maintained or improved.
3. Areas of concern or decline.
4. Context: consider transfer direction (promotion to a higher division means
   maintaining metrics is impressive; relegation to a lower division and improving
   is expected).
5. Overall verdict: did this transfer work based on the data?

Be concise (under 250 words), football-specific, and data-driven."""

    return prompt


def get_llm_analysis(prompt):
    """Call OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            return "⚠️ Set `OPENAI_API_KEY` as environment variable or in `.streamlit/secrets.toml`."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling OpenAI API: {e}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("⚽ Cross-Division Transfer Performance")
    st.caption(
        "Compare player metrics before and after transfers between divisions "
        "(same position only)"
    )

    df, _comps, team_names = load_data()
    country, filtered, player_data = render_sidebar(df, team_names)

    if player_data.empty:
        st.info("👈 Select filters in the sidebar to explore a player.")
        return

    render_header(player_data, country, team_names)
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Radar Overview",
        "⚡ Attacking",
        "🛡️ Defensive",
        "🎯 Passing & Progression",
        "💪 Physical & Engagement",
    ])

    with tab1:
        render_radar(player_data)

    with tab2:
        use_z = st.toggle("Show Z-Scores (league-relative)", key="z_atk")
        render_metric_comparison(player_data, ATTACKING_METRICS, "Attacking", use_z)

    with tab3:
        use_z = st.toggle("Show Z-Scores (league-relative)", key="z_def")
        render_metric_comparison(player_data, DEFENSIVE_METRICS, "Defensive", use_z)

    with tab4:
        use_z = st.toggle("Show Z-Scores (league-relative)", key="z_pass")
        render_metric_comparison(player_data, PASSING_METRICS, "Passing & Progression", use_z)

    with tab5:
        use_z = st.toggle("Show Z-Scores (league-relative)", key="z_phys")
        render_metric_comparison(player_data, PHYSICAL_METRICS, "Physical & Engagement", use_z)

    # ── LLM Analysis ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🤖 Generate AI Analysis", type="primary"):
        with st.spinner("Analyzing with Claude..."):
            prompt = build_llm_prompt(player_data, country, team_names)
            analysis = get_llm_analysis(prompt)
        st.markdown("### 🤖 AI Performance Analysis")
        st.markdown(analysis)


if __name__ == "__main__":
    main()
