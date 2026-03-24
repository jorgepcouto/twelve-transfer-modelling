"""
Streamlit app - Transfer Prediction Explorer

Run:  streamlit run app.py
"""

import os as _os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score

# -- Paths --

_app_dir = _os.path.dirname(_os.path.abspath(__file__))
ACTIVE = _os.path.join(_app_dir, "..", "..", "..", "..", "thesis_data",
                       "processed_data", "thesis_model_dataset", "active")
RAW    = _os.path.join(_app_dir, "..", "..", "..", "..", "thesis_data", "raw_data")

# -- Constants --

QUALITIES = [
    "Active defence", "Aerial threat", "Box threat", "Composure",
    "Defensive heading", "Dribbling", "Effectiveness", "Finishing",
    "Hold-up play", "Intelligent defence", "Involvement",
    "Passing quality", "Pressing", "Progression",
    "Providing teammates", "Run quality", "Winning duels",
]

TEAM_Q = [
    "DEFENCE", "DEFENSIVE_TRANSITION", "ATTACKING_TRANSITION",
    "ATTACK", "PENETRATION", "CHANCE_CREATION", "OUTCOME",
]

TEAM_Q_LABELS = [
    "Defence", "Def. Transition", "Att. Transition",
    "Attack", "Penetration", "Chance Creation", "Outcome",
]

MODEL_SPECS = {
    "M1 - Naive Baseline":          {"id": "1", "delta": False},
    "M2 - Player Profile":          {"id": "2", "delta": False},
    "M3 - Player + Team Context":   {"id": "3", "delta": False},
    "M4 - Tactical Shift":          {"id": "4", "delta": False},
    "M5 - Player + Tactical Shift": {"id": "5", "delta": True},
}

MODEL_NAMES = list(MODEL_SPECS.keys())

DEFAULTS = {
    "Passing quality":      "M3 - Player + Team Context",
    "Involvement":          "M3 - Player + Team Context",
    "Providing teammates":  "M3 - Player + Team Context",
    "Progression":          "M3 - Player + Team Context",
    "Run quality":          "M3 - Player + Team Context",
    "Defensive heading":    "M5 - Player + Tactical Shift",
    "Box threat":           "M3 - Player + Team Context",
    "Pressing":             "M3 - Player + Team Context",
    "Intelligent defence":  "M3 - Player + Team Context",
    "Active defence":       "M5 - Player + Tactical Shift",
    "Effectiveness":        "M3 - Player + Team Context",
    "Aerial threat":        "M3 - Player + Team Context",
    "Winning duels":        "M5 - Player + Tactical Shift",
    "Dribbling":            "M3 - Player + Team Context",
    "Hold-up play":         "M3 - Player + Team Context",
    "Composure":            "M3 - Player + Team Context",
    "Finishing":            "M5 - Player + Tactical Shift",
}

PLAYER_NAMES = {
    260002: "Valentin Rongier", 426712: "Mahdi Camara",
    512649: "Fernando Beltran", 113734: "Jesus Guemez",
    367395: "Samuel Holm", 428282: "Daniel Ask",
    577008: "Nicolai Remberg", 379209: "Declan Rice",
    614501: "Moises Caicedo", 481911: "Alexis Mac Allister",
    21315:  "Jorginho", 69404:  "Mateo Kovacic",
    28292:  "Abdoulaye Doucoure", 25804:  "Moussa Sissoko",
    54:     "Christian Eriksen",
}

# -- Data loading --

@st.cache_data
def load_data():
    df = pd.read_parquet(_os.path.join(ACTIVE, "within_league_transfers_v5.parquet"))
    mf = df[df["from_position"] == "Midfielder"].copy()

    teams = pd.read_parquet(_os.path.join(RAW, "Wyscout", "wyscout_teams.parquet"))
    team_map = dict(zip(teams["team_id"], teams["name"]))

    comps = pd.read_parquet(_os.path.join(RAW, "Wyscout", "competitions_wyscout.parquet"))
    comp_map = dict(zip(comps["competition_id"], comps["name"]))
    country_map = dict(zip(comps["competition_id"], comps["country"]))

    from_pq = [f"from_{q}" for q in QUALITIES]
    to_pq   = [f"to_{q}"   for q in QUALITIES]
    from_tq = [f"from_q_proj_{q}" for q in TEAM_Q]
    to_tq   = [f"to_q_{q}" for q in TEAM_Q]

    for q in TEAM_Q:
        mf[f"delta_team_{q}"] = mf[f"to_q_{q}"] - mf[f"from_q_proj_{q}"]
    delta_tq = [f"delta_team_{q}" for q in TEAM_Q]

    model_cols = from_pq + to_pq + from_tq + to_tq + delta_tq + ["from_season"]
    mf_clean = mf.loc[mf[model_cols].dropna().index].copy()

    mf_clean["_from_team"] = mf_clean["from_team_id"].map(team_map).fillna("Unknown")
    mf_clean["_to_team"]   = mf_clean["to_team_id"].map(team_map).fillna("Unknown")
    mf_clean["_comp"]      = mf_clean["from_competition"].map(comp_map).fillna("Unknown")
    mf_clean["_country"]   = mf_clean["from_competition"].map(country_map).fillna("Unknown")
    mf_clean["_player"]    = mf_clean["wy_player_id"].map(PLAYER_NAMES).fillna("")
    mf_clean["_split"]     = np.where(mf_clean["from_season"] <= 2023, "Train", "Test")
    mf_clean["_label"] = (
        mf_clean["_split"] + " | " +
        mf_clean["_player"].where(mf_clean["_player"] != "", "ID " + mf_clean["wy_player_id"].astype(str)) +
        " | " + mf_clean["_from_team"] + " -> " + mf_clean["_to_team"] +
        " (" + mf_clean["from_season"].astype(str) + ")"
    )

    col_groups = {
        "from_pq": from_pq, "to_pq": to_pq,
        "from_tq": from_tq, "to_tq": to_tq,
        "delta_tq": delta_tq,
    }
    return mf, mf_clean, team_map, col_groups


@st.cache_data
def train_all_models(_mf_clean, col_groups):
    from_pq  = col_groups["from_pq"]
    from_tq  = col_groups["from_tq"]
    to_tq    = col_groups["to_tq"]
    delta_tq = col_groups["delta_tq"]

    train = _mf_clean[_mf_clean["from_season"] <= 2023]
    test  = _mf_clean[_mf_clean["from_season"] == 2024]

    def _features(model_id, q):
        if model_id == "1":
            return [f"from_{q}"]
        elif model_id == "2":
            return list(from_pq)
        elif model_id == "3":
            return list(from_pq) + list(from_tq) + list(to_tq)
        elif model_id == "4":
            return list(delta_tq)
        elif model_id == "5":
            return list(from_pq) + list(delta_tq)

    models = {}
    metrics = []
    test_preds = {}  # (model_name, quality) -> (actual, predicted) arrays

    for mname, spec in MODEL_SPECS.items():
        mid = spec["id"]
        delta = spec["delta"]
        for q in QUALITIES:
            feats = _features(mid, q)
            X_train = sm.add_constant(train[feats])
            if delta:
                y_train = train[f"to_{q}"] - train[f"from_{q}"]
            else:
                y_train = train[f"to_{q}"]
            try:
                result = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
            except Exception:
                continue

            models[(mname, q)] = {"result": result, "feats": feats, "delta": delta}

            if len(test) > 0:
                X_test = sm.add_constant(test[feats])
                pred = result.predict(X_test.astype(float))
                if delta:
                    pred_to = test[f"from_{q}"].values + pred.values
                else:
                    pred_to = pred.values
                actual_to = test[f"to_{q}"].values
                r2  = r2_score(actual_to, pred_to)
                mae = mean_absolute_error(actual_to, pred_to)
                test_preds[(mname, q)] = (actual_to, pred_to)
            else:
                r2, mae = np.nan, np.nan

            metrics.append({
                "Model": mname, "Quality": q,
                "R2_test": r2, "MAE_test": mae,
                "F_pvalue": result.f_pvalue,
            })

    metrics_df = pd.DataFrame(metrics)
    return models, metrics_df, test_preds


# -- Prediction --

def predict_row(row, model_choices, models):
    records = []
    for q in QUALITIES:
        mname = model_choices[q]
        key = (mname, q)
        if key not in models:
            records.append({"Quality": q, "Pre": np.nan, "Predicted": np.nan, "Actual": np.nan})
            continue
        info = models[key]
        feats = info["feats"]
        X = sm.add_constant(
            pd.DataFrame([row[feats].values], columns=feats), has_constant="add",
        )
        pred = info["result"].predict(X.astype(float))[0]
        if info["delta"]:
            pred = float(row[f"from_{q}"]) + pred
        records.append({
            "Quality": q,
            "Pre": float(row[f"from_{q}"]),
            "Predicted": pred,
            "Actual": float(row[f"to_{q}"]),
        })
    out = pd.DataFrame(records).set_index("Quality")
    out["Error"] = out["Predicted"] - out["Actual"]
    return out


# -- Plotly charts --

def fig_team_context(row, team_map):
    from_vals = [float(row[f"from_q_proj_{q}"]) for q in TEAM_Q]
    to_vals   = [float(row[f"to_q_{q}"]) for q in TEAM_Q]
    from_t = team_map.get(row["from_team_id"], "Origin")
    to_t   = team_map.get(row["to_team_id"], "Destination")

    fig = go.Figure()
    for i in range(len(TEAM_Q_LABELS)):
        fig.add_trace(go.Scatter(
            x=[from_vals[i], to_vals[i]], y=[TEAM_Q_LABELS[i], TEAM_Q_LABELS[i]],
            mode="lines", line=dict(color="#bdc3c7", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=from_vals, y=TEAM_Q_LABELS, mode="markers",
        marker=dict(color="#e74c3c", size=12), name=from_t,
    ))
    fig.add_trace(go.Scatter(
        x=to_vals, y=TEAM_Q_LABELS, mode="markers",
        marker=dict(color="#3498db", size=12), name=to_t,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=0.8)
    fig.update_layout(
        title="Team Tactical Qualities (z-score)",
        xaxis_title="z-score", yaxis_title="",
        height=350, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_radars_split(result, visible_qualities):
    """Side-by-side radars: left = pre only, right = actual vs predicted."""
    res = result.loc[[q for q in QUALITIES if q in result.index and q in visible_qualities]]
    if len(res) == 0:
        return go.Figure().update_layout(title="No qualities selected")

    labels = res.index.tolist()
    pre_vals    = res["Pre"].values.tolist()
    pred_vals   = res["Predicted"].values.tolist()
    actual_vals = res["Actual"].values.tolist()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=["Pre-transfer profile", "Predicted vs Actual post-transfer"],
    )

    # Left: pre only
    fig.add_trace(go.Scatterpolar(
        r=pre_vals + [pre_vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(189,195,199,0.15)",
        line=dict(color="#bdc3c7", width=2), name="Pre-transfer",
    ), row=1, col=1)

    # Right: actual + predicted
    fig.add_trace(go.Scatterpolar(
        r=actual_vals + [actual_vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(46,204,113,0.10)",
        line=dict(color="#2ecc71", width=2), name="Actual post",
    ), row=1, col=2)
    fig.add_trace(go.Scatterpolar(
        r=pred_vals + [pred_vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(52,152,219,0.10)",
        line=dict(color="#3498db", width=2, dash="dash"), name="Predicted post",
    ), row=1, col=2)

    fig.update_polars(radialaxis=dict(range=[-2, 2], tickvals=[-1, 0, 1]))
    fig.update_layout(
        height=500, margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def fig_scatter_quality(actual, predicted, quality_name, model_name, r2, mae):
    """Scatter plot actual vs predicted for one quality on test set."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers",
        marker=dict(color="#3498db", size=5, opacity=0.5),
        name="Test observations",
    ))
    lims = [-3, 3]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="grey", width=1, dash="dash"),
        showlegend=False,
    ))
    fig.update_layout(
        title=f"{quality_name}<br><sub>{model_name} | R2={r2:.3f} | MAE={mae:.3f}</sub>",
        xaxis_title="Actual", yaxis_title="Predicted",
        xaxis=dict(range=lims), yaxis=dict(range=lims),
        height=400, width=450,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ====================================================================
#  APP
# ====================================================================

st.set_page_config(page_title="Transfer Prediction Explorer", layout="wide")
st.title("Transfer Prediction Explorer")

mf, mf_clean, team_map, col_groups = load_data()
models, metrics_df, test_preds = train_all_models(mf_clean, col_groups)

tab1, tab2, tab3, tab4 = st.tabs(["Models", "Configuration", "Explorer", "Model Diagnostics"])

# ====================================================================
#  TAB 1 - MODEL INFO
# ====================================================================

with tab1:
    st.header("Model Specifications")
    st.markdown("Five OLS regression models of increasing complexity. All predict **post-transfer quality** for each of the 17 midfielder qualities.")

    st.subheader("M1 - Naive Baseline")
    st.latex(r"\text{to\_Q}_i = \alpha + \beta \cdot \text{from\_Q}_i")
    st.markdown("Simplest hypothesis: post-transfer quality is a linear function of that same quality before the transfer. **1 feature** per quality.")
    st.divider()

    st.subheader("M2 - Player Profile")
    st.latex(r"\text{to\_Q}_i = \alpha + \sum_{j=1}^{17} \beta_j \cdot \text{from\_Q}_j")
    st.markdown("All 17 pre-transfer qualities predict each post-transfer quality. Captures cross-quality effects. **17 features**.")
    st.divider()

    st.subheader("M3 - Player + Team Context")
    st.latex(r"\text{to\_Q}_i = \alpha + \sum_{j=1}^{17} \beta_j \cdot \text{from\_Q}_j + \sum_{k=1}^{7} \gamma_k \cdot \text{from\_TQ}_k + \sum_{k=1}^{7} \delta_k \cdot \text{to\_TQ}_k")
    st.markdown("Adds origin and destination team tactical styles (7 dimensions each). Tests whether team context explains additional variance. **31 features**.")
    st.divider()

    st.subheader("M4 - Tactical Shift")
    st.latex(r"\text{to\_Q}_i = \alpha + \sum_{k=1}^{7} \gamma_k \cdot \Delta \text{TQ}_k")
    st.markdown("Can the change in tactical environment alone explain post-transfer quality? No player baseline -- purely team-driven. **7 features**.")
    st.divider()

    st.subheader("M5 - Player + Tactical Shift")
    st.latex(r"\Delta \text{PQ}_i = \alpha + \sum_{j=1}^{17} \beta_j \cdot \text{from\_Q}_j + \sum_{k=1}^{7} \gamma_k \cdot \Delta \text{TQ}_k")
    st.markdown("Combines player baseline with the change in tactical environment. Predicts **delta**, then reconstructs: `to_Q = from_Q + predicted_delta`. **24 features**.")

# ====================================================================
#  TAB 2 - CONFIGURATION
# ====================================================================

with tab2:
    st.header("Quality Model Configuration")
    st.markdown("For each quality, choose which model to use and toggle visibility in the radar chart.")

    if "model_choices" not in st.session_state:
        st.session_state.model_choices = dict(DEFAULTS)
    if "visible" not in st.session_state:
        st.session_state.visible = {q: True for q in QUALITIES}

    for q in QUALITIES:
        with st.container():
            cols = st.columns([3, 3, 1, 1.5, 1.5, 1.5])
            cols[0].markdown(f"**{q}**")
            chosen = cols[1].selectbox(
                "Model", MODEL_NAMES,
                index=MODEL_NAMES.index(st.session_state.model_choices[q]),
                key=f"cfg_model_{q}", label_visibility="collapsed",
            )
            st.session_state.model_choices[q] = chosen
            on = cols[2].toggle("On", value=st.session_state.visible[q],
                                key=f"cfg_vis_{q}", label_visibility="collapsed")
            st.session_state.visible[q] = on

            row_m = metrics_df[(metrics_df["Model"] == chosen) & (metrics_df["Quality"] == q)]
            if len(row_m) > 0:
                r = row_m.iloc[0]
                cols[3].metric("R2", f"{r['R2_test']:.3f}")
                cols[4].metric("MAE", f"{r['MAE_test']:.3f}")
                pval = r["F_pvalue"]
                cols[5].metric("p-val", f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}")
        st.divider()

# ====================================================================
#  TAB 3 - EXPLORER
# ====================================================================

with tab3:
    st.header("Transfer Explorer")

    model_choices = st.session_state.get("model_choices", dict(DEFAULTS))
    visible = st.session_state.get("visible", {q: True for q in QUALITIES})
    visible_qualities = [q for q in QUALITIES if visible.get(q, True)]

    fc1, fc2, fc3, fc4, fc5 = st.columns([1.2, 1, 2, 2, 2])

    split_filter = fc1.radio("Split", ["All", "Train", "Test"], horizontal=False)
    if split_filter == "Train":
        pool = mf_clean[mf_clean["_split"] == "Train"]
    elif split_filter == "Test":
        pool = mf_clean[mf_clean["_split"] == "Test"]
    else:
        pool = mf_clean

    countries = sorted(pool["_country"].unique())
    sel_country = fc2.selectbox("Country", ["All"] + countries)
    if sel_country != "All":
        pool = pool[pool["_country"] == sel_country]

    from_teams = sorted(pool["_from_team"].unique())
    sel_from = fc3.selectbox("Team from", ["All"] + from_teams)
    if sel_from != "All":
        pool = pool[pool["_from_team"] == sel_from]

    to_teams = sorted(pool["_to_team"].unique())
    sel_to = fc4.selectbox("Team to", ["All"] + to_teams)
    if sel_to != "All":
        pool = pool[pool["_to_team"] == sel_to]

    player_search = fc5.text_input("Player name", "")
    if player_search:
        mask = (
            pool["_player"].str.contains(player_search, case=False, na=False) |
            pool["wy_player_id"].astype(str).str.contains(player_search, na=False)
        )
        pool = pool[mask]

    if len(pool) == 0:
        st.warning("No transfers match the current filters.")
        st.stop()

    labels = pool["_label"].values.tolist()
    selected_label = st.selectbox("Select transfer", labels)
    sel_idx = pool[pool["_label"] == selected_label].index[0]

    row = mf_clean.loc[sel_idx]
    meta = mf.loc[sel_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Split", row["_split"])
    c2.metric("Season", str(int(meta["from_season"])))
    c3.metric("Age", f"{meta['player_season_age']:.1f}")
    fee = meta["tm_transfer_fee"]
    c4.metric("Fee", f"EUR {fee/1e6:.1f}M" if pd.notna(fee) and fee > 0 else "Free / N/A")

    st.markdown(f"**{row['_from_team']}** -> **{row['_to_team']}** &nbsp;|&nbsp; {row['_comp']} ({row['_country']})")

    st.plotly_chart(fig_team_context(meta, team_map), use_container_width=True)

    result = predict_row(mf_clean.loc[sel_idx], model_choices, models)
    st.plotly_chart(fig_radars_split(result, visible_qualities), use_container_width=True)

    st.subheader("Quality-level detail")
    display = result.loc[visible_qualities].copy()
    display["Model"] = [model_choices[q] for q in visible_qualities]
    st.dataframe(
        display[["Pre", "Predicted", "Actual", "Error", "Model"]].style.format(
            {"Pre": "{:.3f}", "Predicted": "{:.3f}", "Actual": "{:.3f}", "Error": "{:+.3f}"}
        ),
        use_container_width=True,
    )
    mae_vis = display["Error"].abs().mean()
    st.metric("MAE (visible qualities)", f"{mae_vis:.3f}")

# ====================================================================
#  TAB 4 - MODEL DIAGNOSTICS
# ====================================================================

with tab4:
    st.header("Model Diagnostics")
    st.markdown("Test-set scatter plots for each quality (sorted by R\u00b2 high \u2192 low). Click a quality below to see the full OLS summary.")

    model_choices = st.session_state.get("model_choices", dict(DEFAULTS))
    visible = st.session_state.get("visible", {q: True for q in QUALITIES})
    visible_qualities = [q for q in QUALITIES if visible.get(q, True)]

    # Gather metrics and sort by R2 descending
    diag_items = []
    for q in visible_qualities:
        mname = model_choices[q]
        key = (mname, q)
        if key not in models or key not in test_preds:
            continue
        row_m = metrics_df[(metrics_df["Model"] == mname) & (metrics_df["Quality"] == q)]
        if len(row_m) == 0:
            continue
        r2 = row_m.iloc[0]["R2_test"]
        mae = row_m.iloc[0]["MAE_test"]
        diag_items.append({"q": q, "mname": mname, "key": key, "r2": r2, "mae": mae})

    diag_items.sort(key=lambda x: x["r2"], reverse=True)

    # Render in a 3-column grid
    n_cols = 3
    for row_start in range(0, len(diag_items), n_cols):
        row_items = diag_items[row_start : row_start + n_cols]
        grid_cols = st.columns(n_cols)
        for col, item in zip(grid_cols, row_items):
            actual, predicted = test_preds[item["key"]]
            fig = fig_scatter_quality(actual, predicted, item["q"], item["mname"],
                                     item["r2"], item["mae"])
            fig.update_layout(height=350, width=None, margin=dict(l=40, r=20, t=60, b=30))
            col.plotly_chart(fig, use_container_width=True)

    # Full summary viewer
    st.divider()
    st.subheader("Full Regression Summary")
    summary_options = [f"{item['q']}  (R\u00b2={item['r2']:.3f})" for item in diag_items]
    if summary_options:
        sel_summary = st.selectbox("Select quality", summary_options)
        sel_idx_diag = summary_options.index(sel_summary)
        item = diag_items[sel_idx_diag]
        ols_result = models[item["key"]]["result"]
        st.code(ols_result.summary().as_text(), language=None)
