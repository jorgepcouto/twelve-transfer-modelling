"""
Transfer Performance Lab
========================
Position-aware player performance analysis across league transfers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Transfer Performance Lab", page_icon="⚽", layout="wide")

# ── Dark theme CSS (Opta-style) ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Header */
.player-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
    border-left: 4px solid #e94560; color: #eee;
}
.player-card h1 { color: #fff; margin: 0 0 4px 0; font-size: 2rem; font-weight: 700; }
.player-card .position-badge {
    display: inline-block; background: #e94560; color: #fff;
    padding: 3px 14px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 600; margin-left: 10px; vertical-align: middle;
}
.player-card .transfer-route {
    color: #a0aec0; font-size: 0.95rem; margin-top: 8px;
}
.player-card .transfer-route b { color: #e2e8f0; }

/* KPI cards */
.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin: 20px 0;
}
.kpi-card {
    background: #0f3460; border-radius: 10px; padding: 16px;
    text-align: center; border: 1px solid #1a1a4e;
}
.kpi-card .label { color: #8899aa; font-size: 0.7rem; text-transform: uppercase;
    font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; }
.kpi-card .value { color: #fff; font-size: 1.4rem; font-weight: 700; }
.kpi-card .delta { font-size: 0.8rem; font-weight: 600; margin-top: 4px; }
.delta-up { color: #48bb78; }
.delta-down { color: #fc8181; }
.delta-neutral { color: #a0aec0; }

/* Section headers */
.section-header {
    color: #e94560; font-size: 0.85rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 32px 0 16px 0; padding-bottom: 8px;
    border-bottom: 2px solid #e94560;
}

/* AI card */
.ai-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 12px; padding: 24px; margin-top: 24px;
    border: 1px solid #1a1a4e;
}
.ai-card h3 { color: #e94560; margin-top: 0; font-size: 1rem; }
.ai-card p, .ai-card li { color: #cbd5e0; line-height: 1.7; }

/* Metric detail cards */
.metric-row {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px; margin: 12px 0;
}
.metric-item {
    background: #16213e; border-radius: 8px; padding: 12px 14px;
    border: 1px solid #1a1a4e;
}
.metric-item .m-label { color: #8899aa; font-size: 0.72rem; font-weight: 500; }
.metric-item .m-vals {
    display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;
}
.metric-item .m-before { color: #63b3ed; font-size: 0.95rem; font-weight: 600; }
.metric-item .m-after { color: #fc8181; font-size: 0.95rem; font-weight: 600; }
.metric-item .m-arrow { color: #718096; font-size: 0.8rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0a0a1a; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label { color: #a0aec0 !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Position Templates ───────────────────────────────────────────────────────

POSITION_CONFIG = {
    "Goalkeeper": {
        "identity": [
            ("Commanding", "Defensive aerials won per 90"),
            ("Shot Stopping", "Defending 1v1 %"),
            ("Distribution", "Passes (xT) per 90"),
            ("Sweeping", "High recoveries per 90"),
            ("Ball Playing", "Ball progression (xT) per 90"),
            ("Composure", "Composure"),
            ("Involvement", "Involvement"),
        ],
        "detail_metrics": [
            "Defensive actions per 90", "Defensive aerials won %",
            "Interceptions per 90", "Ball recoveries per 90",
            "Passes (xT) per 90", "Touches per 90",
            "Long ball receptions per 90", "Pressure resistance %",
            "Losses per 90",
        ],
        "llm_focus": "shot stopping, distribution quality, sweeping/claiming ability, and composure under pressure",
    },
    "Central Defender": {
        "identity": [
            ("Aerial Dom.", "Aerials won per 90"),
            ("Tackling", "True tackles won per 90"),
            ("Progression", "Ball progression (xT) per 90"),
            ("Passing", "Passing quality"),
            ("Defensive IQ", "Intelligent defence"),
            ("Duels", "Winning duels"),
            ("Heading", "Defensive heading"),
        ],
        "detail_metrics": [
            "Aerials won %", "Defensive duels won %", "Tackle success %",
            "Interceptions per 90", "Defensive actions per 90",
            "Passes (xT) per 90", "Deep completions per 90",
            "Carries (xT) per 90", "Pressure resistance %",
            "Losses per 90", "Possessions won per 90",
            "Counterpressing per 90",
        ],
        "llm_focus": "aerial dominance, tackling, ball progression from the back, and defensive reading of the game",
    },
    "Full Back": {
        "identity": [
            ("Offensive Output", "xA per 90"),
            ("Crossing", "Crosses (xT) per 90"),
            ("Progression", "Progression"),
            ("Active Defence", "Active defence"),
            ("1v1 Defending", "Defending 1v1 %"),
            ("Involvement", "Involvement"),
            ("Dribbling", "Dribbling"),
        ],
        "detail_metrics": [
            "xA per 90", "Key passes per 90", "Crosses (xT) per 90",
            "Deep completions per 90", "Deep runs (xT) per 90",
            "Ball progression (xT) per 90", "Carries (xT) per 90",
            "Defensive duels won per 90", "Interceptions per 90",
            "Touches per 90", "Pressure resistance %",
        ],
        "llm_focus": "offensive contribution (crosses, xA), progressive carrying, and defensive reliability in 1v1 situations",
    },
    "Midfielder": {
        "identity": [
            ("Creativity", "xGCreated per 90"),
            ("Progression", "Ball progression (xT) per 90"),
            ("Passing", "Passing quality"),
            ("Pressing", "Pressing"),
            ("Composure", "Composure"),
            ("Involvement", "Involvement"),
            ("Providing", "Providing teammates"),
        ],
        "detail_metrics": [
            "xA per 90", "Assists per 90", "Key passes per 90",
            "Creative passes per 90", "Playmaking passes per 90",
            "Passes (xT) per 90", "Deep completions per 90",
            "Ball progression (xT) per 90", "Carries (xT) per 90",
            "Counterpressing per 90", "Possessions won per 90",
            "Ball recoveries per 90", "xGBuildup per 90",
            "xGChain per possession", "Pressure resistance %",
            "Touches per 90",
        ],
        "llm_focus": "creativity, ball progression, pressing intensity, game control, and ability to provide passing options",
    },
    "Winger": {
        "identity": [
            ("Goal Threat", "xG per 90"),
            ("Creativity", "xA per 90"),
            ("Dribbling", "Dribbling"),
            ("Run Quality", "Run quality"),
            ("Box Presence", "Box threat"),
            ("Involvement", "Involvement"),
            ("Providing", "Providing teammates"),
        ],
        "detail_metrics": [
            "xG per 90", "Goals per 90", "xGOT per 90", "Shot conversion %",
            "xA per 90", "Key passes per 90", "Crosses (xT) per 90",
            "Dribbles (xT) per 90", "Dribbles success %",
            "Successful 1v1 per 90", "Deep runs (xT) per 90",
            "Box entries per 90", "Touches in box per 90",
            "Counterpressing per 90",
        ],
        "llm_focus": "goal threat (xG, finishing), creativity (xA, key passes), dribbling ability, and movement quality into the box",
    },
    "Striker": {
        "identity": [
            ("Finishing", "Finishing"),
            ("Goal Threat", "xG per 90"),
            ("Poaching", "Poaching"),
            ("Hold-up", "Hold-up play"),
            ("Box Presence", "Box threat"),
            ("Aerial Threat", "Aerial threat"),
            ("Run Quality", "Run quality"),
        ],
        "detail_metrics": [
            "xG per 90", "Goals per 90", "xGOT per 90", "Goals - xG",
            "Shot conversion %", "xG per shot", "Goals per box touch",
            "Touches in box per 90", "Box entries per 90",
            "Headed plays per 90", "Aerials won per 90",
            "xA per 90", "Linkups per 90",
            "Dribbles (xT) per 90",
        ],
        "llm_focus": "finishing quality, movement and poaching, hold-up play, aerial presence, and link-up play",
    },
}

RADAR_METRICS = [
    "Hold-up play", "Involvement", "Providing teammates", "Aerial threat",
    "Poaching", "Run quality", "Pressing", "Finishing", "Box threat",
    "Dribbling", "Active defence", "Progression", "Intelligent defence",
    "Defensive heading", "Passing quality", "Winning duels", "Composure",
    "Effectiveness", "Territorial dominance", "Chance prevention",
]

# ── Colors ───────────────────────────────────────────────────────────────────

C_BEFORE = "#63b3ed"   # blue
C_AFTER = "#e94560"    # red/coral
C_BG = "#1a1a2e"
C_CARD = "#16213e"

# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path(__file__).parent
    df = pd.read_parquet(base / "../../thesis_data/raw_data_twelve/Twelve/male_transfer_model.parquet")
    comps = pd.read_parquet(base / "../../thesis_data/raw_data_twelve/Wyscout/competitions_wyscout.parquet")

    # Competition lookups
    comp_meta = comps.drop_duplicates("competition_id").set_index("competition_id")
    comp_to_name = comp_meta["name"].to_dict()
    comp_to_country = comp_meta["country"].to_dict()

    # Team name lookup
    team_names = {}
    clean_path = base / "../../thesis_data/processed/transfers_clean.parquet"
    if clean_path.exists():
        clean = pd.read_parquet(clean_path, columns=["from_team_id", "to_team_id", "tm_team_from", "tm_team_to"])
        for _, row in clean.dropna(subset=["tm_team_from"]).iterrows():
            team_names[int(row["from_team_id"])] = row["tm_team_from"]
        for _, row in clean.dropna(subset=["tm_team_to"]).iterrows():
            team_names[int(row["to_team_id"])] = row["tm_team_to"]

    # Pre-filter: only cross-league, same position
    df = df[
        (df["from_competition"] != df["to_competition"])
        & (df["from_position"] == df["to_position"])
    ].copy()

    # Add readable names
    df["from_league"] = df["from_competition"].map(comp_to_name).fillna("Unknown")
    df["to_league"] = df["to_competition"].map(comp_to_name).fillna("Unknown")
    df["from_country"] = df["from_competition"].map(comp_to_country).fillna("Unknown")
    df["to_country"] = df["to_competition"].map(comp_to_country).fillna("Unknown")

    return df, team_names


def tname(team_id, team_names):
    return team_names.get(int(team_id), f"Team {team_id}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df, team_names):
    st.sidebar.markdown("### ⚽ Transfer Performance Lab")
    st.sidebar.caption("Filter to explore player performance across league transfers")
    st.sidebar.divider()

    filtered = df.copy()

    # 1. From Country
    countries_from = sorted(filtered["from_country"].unique())
    sel_from_country = st.sidebar.selectbox("From Country", ["All"] + countries_from)
    if sel_from_country != "All":
        filtered = filtered[filtered["from_country"] == sel_from_country]

    # 2. From League
    from_leagues = sorted(filtered["from_league"].unique())
    sel_from_league = st.sidebar.selectbox("From League", ["All"] + from_leagues)
    if sel_from_league != "All":
        filtered = filtered[filtered["from_league"] == sel_from_league]

    # 3. To Country
    countries_to = sorted(filtered["to_country"].unique())
    sel_to_country = st.sidebar.selectbox("To Country", ["All"] + countries_to)
    if sel_to_country != "All":
        filtered = filtered[filtered["to_country"] == sel_to_country]

    # 4. To League
    to_leagues = sorted(filtered["to_league"].unique())
    sel_to_league = st.sidebar.selectbox("To League", ["All"] + to_leagues)
    if sel_to_league != "All":
        filtered = filtered[filtered["to_league"] == sel_to_league]

    # 5. Promo/releg toggle
    include_promo = st.sidebar.checkbox("Include promotion/relegation (same team)", value=True)
    if not include_promo:
        filtered = filtered[filtered["from_team_id"] != filtered["to_team_id"]]

    if filtered.empty:
        st.sidebar.warning("No transfers match filters.")
        return pd.DataFrame()

    # 6. Season
    seasons = sorted(filtered["to_season"].unique())
    sel_season = st.sidebar.selectbox("Season", ["All"] + [str(s) for s in seasons])
    if sel_season != "All":
        filtered = filtered[filtered["to_season"] == int(sel_season)]

    # 7. Position
    positions = sorted(filtered["from_position"].unique())
    sel_pos = st.sidebar.selectbox("Position", ["All"] + positions)
    if sel_pos != "All":
        filtered = filtered[filtered["from_position"] == sel_pos]

    if filtered.empty:
        st.sidebar.warning("No transfers match filters.")
        return pd.DataFrame()

    # 8. Team
    team_ids = sorted(filtered["to_team_id"].unique())
    team_opts = {tid: tname(tid, team_names) for tid in team_ids}
    sel_team = st.sidebar.selectbox(
        "Destination Team", ["All"] + team_ids,
        format_func=lambda x: "All" if x == "All" else team_opts.get(x, str(x)),
    )
    if sel_team != "All":
        filtered = filtered[filtered["to_team_id"] == sel_team]

    if filtered.empty:
        st.sidebar.warning("No transfers match filters.")
        return pd.DataFrame()

    # 9. Player
    players = filtered[["player_id", "short_name"]].drop_duplicates().sort_values("short_name")
    player_opts = dict(zip(players["player_id"], players["short_name"]))
    sel_player = st.sidebar.selectbox("Player", list(player_opts.keys()), format_func=lambda x: player_opts[x])

    player_data = filtered[filtered["player_id"] == sel_player]

    # Multiple transfers
    if len(player_data) > 1:
        labels = [f"{int(r['from_season'])} → {int(r['to_season'])} | {r['from_league']} → {r['to_league']}" for _, r in player_data.iterrows()]
        idx = st.sidebar.selectbox("Transfer", range(len(labels)), format_func=lambda i: labels[i])
        player_data = player_data.iloc[[idx]]

    st.sidebar.divider()
    st.sidebar.caption(f"📊 {len(filtered):,} records · {filtered['player_id'].nunique():,} players")

    return player_data


# ── Player Card (HTML) ───────────────────────────────────────────────────────

def render_player_card(r, team_names):
    pos = r["from_position"]
    from_team = tname(r["from_team_id"], team_names)
    to_team = tname(r["to_team_id"], team_names)
    is_promo = int(r["from_team_id"]) == int(r["to_team_id"])
    transfer_tag = "PROMOTION / RELEGATION" if is_promo else "TRANSFER"
    mins_before = f"{r['from_Minutes']:.0f}"
    mins_after = f"{r['to_Minutes']:.0f}" if pd.notna(r.get("to_Minutes")) else "—"

    st.markdown(f"""
    <div class="player-card">
        <h1>{r['short_name']} <span class="position-badge">{pos}</span></h1>
        <div class="transfer-route">
            <b>{r['from_league']}</b> ({from_team}) → <b>{r['to_league']}</b> ({to_team})
            &nbsp;·&nbsp; {transfer_tag}
        </div>
        <div class="transfer-route" style="margin-top:6px;">
            Season <b>{int(r['from_season'])}</b> → <b>{int(r['to_season'])}</b>
            &nbsp;·&nbsp; Minutes: <b>{mins_before}</b> → <b>{mins_after}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── KPI Identity Vector ─────────────────────────────────────────────────────

def render_identity_vector(r, pos):
    config = POSITION_CONFIG.get(pos)
    if not config:
        return

    st.markdown('<div class="section-header">Player Identity Vector</div>', unsafe_allow_html=True)

    cards_html = '<div class="kpi-grid">'
    for label, metric in config["identity"]:
        fv = r.get(f"from_{metric}")
        tv = r.get(f"to_{metric}")
        if pd.notna(fv) and pd.notna(tv):
            fv, tv = float(fv), float(tv)
            delta = tv - fv
            delta_cls = "delta-up" if delta > 0.05 else ("delta-down" if delta < -0.05 else "delta-neutral")
            delta_arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "—")
            cards_html += f"""
            <div class="kpi-card">
                <div class="label">{label}</div>
                <div class="value">{tv:.2f}</div>
                <div class="delta {delta_cls}">{delta_arrow} {delta:+.2f}</div>
            </div>"""
        else:
            cards_html += f"""
            <div class="kpi-card">
                <div class="label">{label}</div>
                <div class="value">—</div>
                <div class="delta delta-neutral">N/A</div>
            </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ── Radar Chart ──────────────────────────────────────────────────────────────

def render_radar(r):
    labels, from_vals, to_vals = [], [], []
    for m in RADAR_METRICS:
        fv, tv = r.get(f"from_{m}"), r.get(f"to_{m}")
        if pd.notna(fv) and pd.notna(tv):
            labels.append(m)
            from_vals.append(float(fv))
            to_vals.append(float(tv))

    if not labels:
        return

    st.markdown('<div class="section-header">Composite Performance Radar</div>', unsafe_allow_html=True)

    labels_c = labels + [labels[0]]
    from_c = from_vals + [from_vals[0]]
    to_c = to_vals + [to_vals[0]]
    rng = max(abs(min(from_vals + to_vals)), abs(max(from_vals + to_vals))) * 1.15

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=from_c, theta=labels_c, fill="toself", name="Before",
        opacity=0.45, line=dict(color=C_BEFORE, width=2),
        fillcolor="rgba(99,179,237,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=to_c, theta=labels_c, fill="toself", name="After",
        opacity=0.6, line=dict(color=C_AFTER, width=2),
        fillcolor="rgba(233,69,96,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0f0f23",
            radialaxis=dict(visible=True, range=[-rng, rng], gridcolor="#1a1a4e",
                            tickfont=dict(color="#556", size=9)),
            angularaxis=dict(gridcolor="#1a1a4e", tickfont=dict(color="#a0aec0", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=12)),
        height=520, margin=dict(t=30, b=60, l=60, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Detail Metrics ───────────────────────────────────────────────────────────

def render_detail_metrics(r, pos):
    config = POSITION_CONFIG.get(pos)
    if not config:
        return

    metrics = config["detail_metrics"]

    st.markdown('<div class="section-header">Detailed Metrics</div>', unsafe_allow_html=True)

    use_z = st.toggle("Show Z-Scores (league-relative)", key="z_detail")

    # Collect data
    data = []
    for m in metrics:
        col = f"z_score_{m}" if use_z else m
        fv, tv = r.get(f"from_{col}"), r.get(f"to_{col}")
        if pd.notna(fv) and pd.notna(tv):
            data.append({"metric": m, "before": float(fv), "after": float(tv), "delta": float(tv) - float(fv)})

    if not data:
        st.info("No detail metrics available.")
        return

    mdf = pd.DataFrame(data)

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before", x=mdf["metric"], y=mdf["before"],
        marker_color=C_BEFORE, marker_line=dict(width=0), opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="After", x=mdf["metric"], y=mdf["after"],
        marker_color=C_AFTER, marker_line=dict(width=0), opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0", size=11),
        xaxis=dict(tickangle=-45, gridcolor="#1a1a4e", tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#1a1a4e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=420, margin=dict(b=140, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metric cards grid
    suffix = " (z)" if use_z else ""
    cards_html = '<div class="metric-row">'
    for d in data:
        delta_cls = "delta-up" if d["delta"] > 0.01 else ("delta-down" if d["delta"] < -0.01 else "delta-neutral")
        cards_html += f"""
        <div class="metric-item">
            <div class="m-label">{d['metric']}{suffix}</div>
            <div class="m-vals">
                <span class="m-before">{d['before']:.2f}</span>
                <span class="m-arrow">→</span>
                <span class="m-after">{d['after']:.2f}</span>
            </div>
            <div class="delta {delta_cls}" style="font-size:0.75rem; margin-top:2px;">Δ {d['delta']:+.3f}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ── LLM Analysis ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_llm_analysis(player_id, from_comp, to_comp, from_season, to_season, pos, prompt):
    """Cached LLM call keyed by player+transfer combination."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {e}"


def build_prompt(r, pos, team_names):
    config = POSITION_CONFIG.get(pos, {})
    focus = config.get("llm_focus", "overall performance")
    from_team = tname(r["from_team_id"], team_names)
    to_team = tname(r["to_team_id"], team_names)

    # Identity vector
    identity_lines = []
    for label, metric in config.get("identity", []):
        fv, tv = r.get(f"from_{metric}"), r.get(f"to_{metric}")
        if pd.notna(fv) and pd.notna(tv):
            identity_lines.append(f"  {label} ({metric}): {fv:.2f} → {tv:.2f} (Δ {tv - fv:+.2f})")

    # Detail metrics: top movers
    detail_changes = []
    for m in config.get("detail_metrics", []):
        fv, tv = r.get(f"from_{m}"), r.get(f"to_{m}")
        if pd.notna(fv) and pd.notna(tv) and fv != 0:
            pct = (tv - fv) / abs(fv) * 100
            detail_changes.append((m, float(fv), float(tv), pct))
    detail_changes.sort(key=lambda x: x[3], reverse=True)

    prompt = f"""You are an elite football performance analyst. Analyze this {pos}'s
performance change after a cross-league transfer. Answer in Spanish.

PLAYER: {r['short_name']}
POSITION: {pos}
TRANSFER: {r['from_league']} ({from_team}) → {r['to_league']} ({to_team})
SEASONS: {r['from_season']} → {r['to_season']}
MINUTES: {r['from_Minutes']:.0f} → {r.get('to_Minutes', 0):.0f}

IDENTITY VECTOR (key {pos} KPIs):
{chr(10).join(identity_lines)}

TOP IMPROVEMENTS:
{chr(10).join(f'  {m[0]}: {m[1]:.2f} → {m[2]:.2f} ({m[3]:+.1f}%)' for m in detail_changes[:5])}

TOP DECLINES:
{chr(10).join(f'  {m[0]}: {m[1]:.2f} → {m[2]:.2f} ({m[3]:+.1f}%)' for m in detail_changes[-5:])}

FOCUS YOUR ANALYSIS ON: {focus}

Provide:
1. Executive summary (2 sentences max).
2. Key strengths for a {pos} that improved or held.
3. Concerns specific to the {pos} role.
4. Context: moving to a higher-level league and maintaining metrics is impressive;
   moving down and improving is expected. Factor this in.
5. One-line verdict.

Be concise (under 200 words), data-driven, position-specific."""

    return prompt


def render_ai_analysis(r, pos, team_names):
    st.markdown('<div class="section-header">AI Performance Analysis</div>', unsafe_allow_html=True)

    prompt = build_prompt(r, pos, team_names)
    with st.spinner("Generating analysis..."):
        analysis = get_llm_analysis(
            int(r["player_id"]), int(r["from_competition"]), int(r["to_competition"]),
            int(r["from_season"]), int(r["to_season"]), pos, prompt,
        )

    if analysis is None:
        st.markdown("""
        <div class="ai-card">
            <h3>⚠️ API Key Required</h3>
            <p>Set <code>OPENAI_API_KEY</code> as environment variable to enable AI analysis.</p>
        </div>""", unsafe_allow_html=True)
    elif analysis.startswith("⚠️"):
        st.markdown(f"""
        <div class="ai-card"><h3>⚠️ Error</h3><p>{analysis}</p></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-card">
            <h3>🤖 AI Analysis · GPT-4o-mini</h3>
            <p>{analysis}</p>
        </div>""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df, team_names = load_data()
    player_data = render_sidebar(df, team_names)

    if player_data.empty:
        st.markdown("""
        <div style="text-align:center; padding:80px 20px; color:#556;">
            <h2 style="color:#e94560;">Transfer Performance Lab</h2>
            <p style="font-size:1.1rem;">Select filters in the sidebar to explore a player's
            performance before and after a cross-league transfer.</p>
            <p style="color:#445; font-size:0.9rem;">62,578 transfers · 38,767 players · 327 competitions</p>
        </div>
        """, unsafe_allow_html=True)
        return

    r = player_data.iloc[0]
    pos = r["from_position"]

    # 1. Player card
    render_player_card(r, team_names)

    # 2. Identity vector
    render_identity_vector(r, pos)

    # 3. Radar
    render_radar(r)

    # 4. Detail metrics
    render_detail_metrics(r, pos)

    # 5. AI Analysis (automatic)
    render_ai_analysis(r, pos, team_names)


if __name__ == "__main__":
    main()
