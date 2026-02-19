"""
dashboard.py
Real-Time Bridge Anomaly Detection Dashboard — Streamlit + Plotly

Launch:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

# Database: 1) Streamlit secrets, 2) DATABASE_URL env var, 3) localhost fallback
try:
    DB_URL = st.secrets["database"]["url"]
except (KeyError, FileNotFoundError):
    DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/timeseries")
REFRESH_INTERVAL = 5         # seconds
TIME_WINDOW_MINUTES = 30     # show last N minutes of data on charts

# Color palette
C = {
    "bg":         "#0E1117",
    "card":       "#1A1D23",
    "accent":     "#00D4FF",
    "accent2":    "#7C3AED",
    "danger":     "#FF4B4B",
    "success":    "#00C853",
    "warning":    "#FFB300",
    "text":       "#FAFAFA",
    "muted":      "#8B8D97",
    "grid":       "#1E2230",
    "teal":       "#36D7B7",
}


# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bridge Anomaly Detection",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────── */
    .stApp {
        background: #0E1117;
        font-family: 'Inter', sans-serif;
    }

    /* ── Metric Cards ───────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1A1D23, #1E2230);
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,255,0.05);
    }
    div[data-testid="stMetric"] label {
        color: #8B8D97 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.7rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FAFAFA !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
    }

    /* ── Chart Cards ────────────────────────────────────── */
    .chart-card {
        background: linear-gradient(145deg, #1A1D23, #1E2230);
        border: 1px solid rgba(0, 212, 255, 0.08);
        border-radius: 16px;
        padding: 20px 20px 12px 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .chart-title {
        color: #FAFAFA;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin: 0 0 12px 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .chart-title .icon {
        font-size: 1rem;
    }

    /* ── Dataframe ──────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Header ─────────────────────────────────────────── */
    .dash-header {
        background: linear-gradient(145deg, rgba(0,212,255,0.06), rgba(124,58,237,0.06));
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .dash-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .dash-sub {
        color: #8B8D97;
        font-size: 0.82rem;
        margin: 2px 0 0 0;
    }
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,200,83,0.1);
        border: 1px solid rgba(0,200,83,0.25);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #00C853;
    }
    .live-dot {
        width: 7px; height: 7px;
        background: #00C853;
        border-radius: 50%;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── Divider ────────────────────────────────────────── */
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent); margin: 16px 0; }

    /* ── Hide defaults ──────────────────────────────────── */
    #MainMenu, header, footer, div[data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATABASE QUERIES
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


def fetch_latest(engine, limit=500):
    """Fetch the latest scored readings for charts."""
    q = text(f"""
        SELECT id, timestamp,
               strain_microstrain, deflection_mm, vibration_ms2,
               tilt_deg, displacement_mm, cable_member_tension_kn,
               vehicle_load_tons, wind_speed_ms, temperature_c,
               structural_health_index_shi,
               is_anomaly, anomaly_score, bridge_mood_meter
        FROM bridge_dataset
        WHERE is_anomaly IS NOT NULL
        ORDER BY id DESC
        LIMIT {limit}
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    return df.sort_values("id").reset_index(drop=True)


def fetch_kpis(engine):
    """Aggregate KPIs."""
    q = text("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_anomaly = TRUE THEN 1 ELSE 0 END) as anomalies,
            AVG(structural_health_index_shi) as avg_shi,
            COUNT(CASE WHEN is_anomaly IS NULL THEN 1 END) as pending
        FROM bridge_dataset
    """)
    with engine.connect() as conn:
        r = conn.execute(q).fetchone()
    scored = r[0] - (r[4] if len(r) > 4 else r[3])
    return {
        "total": r[0] or 0,
        "anomalies": int(r[1] or 0),
        "avg_shi": float(r[2] or 0),
        "pending": int(r[3] or 0),
        "scored": r[0] - int(r[3] or 0),
    }


def fetch_anomaly_table(engine, limit=12):
    """Recent anomalies for the table."""
    q = text(f"""
        SELECT id, timestamp, strain_microstrain, vibration_ms2,
               cable_member_tension_kn, vehicle_load_tons, anomaly_score
        FROM bridge_dataset
        WHERE is_anomaly = TRUE
        ORDER BY id DESC
        LIMIT {limit}
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn)


# ═══════════════════════════════════════════════════════════════
#  PLOTLY SHARED SETTINGS
# ═══════════════════════════════════════════════════════════════

def base_layout(height=320):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["muted"], size=11),
        margin=dict(l=50, r=20, t=12, b=40),
        height=height,
        xaxis=dict(
            gridcolor=C["grid"], gridwidth=1,
            zeroline=False, showline=False,
            tickfont=dict(size=9),
            tickformat="%d %b\n%H:%M",
            nticks=8,
        ),
        yaxis=dict(
            gridcolor=C["grid"], gridwidth=1,
            zeroline=False, showline=False,
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C["muted"]),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hoverlabel=dict(
            bgcolor="#1E2230", font_size=12, font_color=C["text"],
            bordercolor="rgba(0,212,255,0.3)",
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════

def chart_strain(df):
    """Strain trend — clean line with highlighted anomaly markers."""
    fig = go.Figure()

    # Main line (all data, continuous)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["strain_microstrain"],
        mode="lines",
        name="Strain",
        line=dict(color=C["accent"], width=1.5),
        opacity=0.85,
        hovertemplate="%{y:.0f} µε<extra></extra>",
    ))

    # Anomaly overlay
    anom = df[df["is_anomaly"] == True]
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom["strain_microstrain"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=C["danger"], size=7, symbol="circle",
                line=dict(color="rgba(255,255,255,0.6)", width=1.5),
            ),
            hovertemplate="<b>⚠ %{y:.0f} µε</b><extra></extra>",
        ))

    fig.update_layout(**base_layout(300))
    return fig


def chart_correlation(df):
    """Vehicle load vs deflection — density scatter."""
    fig = go.Figure()

    normal = df[df["is_anomaly"] == False]
    anom = df[df["is_anomaly"] == True]

    fig.add_trace(go.Scatter(
        x=normal["vehicle_load_tons"], y=normal["deflection_mm"],
        mode="markers", name="Normal",
        marker=dict(color=C["accent"], size=4, opacity=0.35),
        hovertemplate="Load: %{x:.0f}t<br>Defl: %{y:.1f}mm<extra></extra>",
    ))

    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["vehicle_load_tons"], y=anom["deflection_mm"],
            mode="markers", name="Anomaly",
            marker=dict(
                color=C["danger"], size=8, opacity=0.85,
                line=dict(color="white", width=1),
            ),
            hovertemplate="<b>⚠</b> Load: %{x:.0f}t<br>Defl: %{y:.1f}mm<extra></extra>",
        ))

    fig.update_layout(**base_layout(300))
    fig.update_xaxes(title_text="Vehicle Load (tons)", title_font=dict(size=10, color=C["muted"]))
    fig.update_yaxes(title_text="Deflection (mm)", title_font=dict(size=10, color=C["muted"]))
    return fig


def chart_anomaly_scores(df):
    """Anomaly scores as a bar/stem chart — much cleaner than area fill."""
    scored = df[df["anomaly_score"].notna()].copy()
    if len(scored) == 0:
        return go.Figure().update_layout(**base_layout(260))

    # Color bars by anomaly status
    colors = [C["danger"] if a else C["accent2"] for a in scored["is_anomaly"]]
    alphas = [0.9 if a else 0.4 for a in scored["is_anomaly"]]

    fig = go.Figure()

    # Bars for scores
    fig.add_trace(go.Bar(
        x=scored["timestamp"], y=scored["anomaly_score"],
        marker=dict(color=colors, opacity=alphas, line=dict(width=0)),
        name="Score",
        hovertemplate="Score: %{y:.4f}<extra></extra>",
    ))

    # Threshold line at 0
    fig.add_hline(
        y=0, line_dash="dot", line_color=C["warning"], line_width=1,
        annotation_text="  threshold", annotation_font_color=C["warning"],
        annotation_font_size=9, annotation_position="right",
    )

    layout = base_layout(260)
    layout["bargap"] = 0
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Decision Score", title_font=dict(size=10, color=C["muted"]))
    return fig


def chart_shi_gauge(shi_value):
    """SHI gauge — cleaner with larger number."""
    if shi_value >= 0.8:
        bar_color = C["success"]
        status = "Healthy"
    elif shi_value >= 0.6:
        bar_color = C["warning"]
        status = "Warning"
    else:
        bar_color = C["danger"]
        status = "Critical"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=shi_value,
        number=dict(font=dict(size=48, color=C["text"], family="Inter"), valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=C["muted"], size=10), dtick=0.2),
            bar=dict(color=bar_color, thickness=0.65),
            bgcolor=C["grid"],
            borderwidth=0,
            steps=[
                dict(range=[0, 0.4], color="rgba(255,75,75,0.12)"),
                dict(range=[0.4, 0.7], color="rgba(255,179,0,0.08)"),
                dict(range=[0.7, 1], color="rgba(0,200,83,0.06)"),
            ],
            threshold=dict(line=dict(color=C["danger"], width=2), thickness=0.75, value=0.6),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"]),
        margin=dict(l=28, r=28, t=28, b=0),
        height=200,
    )
    return fig, status


def chart_vibration(df):
    """Vibration — clean teal line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["vibration_ms2"],
        mode="lines", name="Vibration",
        line=dict(color=C["teal"], width=1.5),
        opacity=0.8,
        hovertemplate="%{y:.3f} m/s²<extra></extra>",
    ))

    anom = df[df["is_anomaly"] == True]
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom["vibration_ms2"],
            mode="markers", name="Anomaly",
            marker=dict(color=C["danger"], size=7, line=dict(color="white", width=1.5)),
            hovertemplate="<b>⚠ %{y:.3f} m/s²</b><extra></extra>",
        ))

    fig.update_layout(**base_layout(260))
    return fig


def chart_tension(df):
    """Cable tension — subtle purple line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cable_member_tension_kn"],
        mode="lines", name="Tension",
        line=dict(color=C["accent2"], width=1.5),
        opacity=0.75,
        hovertemplate="%{y:.0f} kN<extra></extra>",
    ))

    anom = df[df["is_anomaly"] == True]
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom["cable_member_tension_kn"],
            mode="markers", name="Anomaly",
            marker=dict(color=C["danger"], size=7, line=dict(color="white", width=1.5)),
            hovertemplate="<b>⚠ %{y:.0f} kN</b><extra></extra>",
        ))

    fig.update_layout(**base_layout(260))
    fig.update_yaxes(title_text="Tension (kN)", title_font=dict(size=10, color=C["muted"]))
    return fig


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <div>
        <p class="dash-title">🌉 Bridge Anomaly Detection</p>
        <p class="dash-sub">Real-time structural health monitoring · IsolationForest ML Engine</p>
    </div>
    <div class="live-pill">
        <div class="live-dot"></div>
        LIVE
    </div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=REFRESH_INTERVAL)
def live_dashboard():
    engine = get_engine()

    try:
        kpis = fetch_kpis(engine)
        df = fetch_latest(engine)
    except Exception as e:
        st.error(f"⚠ Database connection error: {e}")
        st.info("Ensure PostgreSQL is running and data has been seeded.")
        return

    if kpis["total"] == 0:
        st.warning("No data in the database. Run `python3 seed_historical_data.py` first.")
        return

    # ── KPI Row ───────────────────────────────────────────
    scored = kpis["scored"]
    rate = (kpis["anomalies"] / scored * 100) if scored > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Readings", f"{kpis['total']:,}", delta=f"{scored:,} scored", delta_color="off")
    k2.metric("Anomalies Found", f"{kpis['anomalies']:,}", delta=f"{rate:.1f}% rate", delta_color="inverse" if rate > 5 else "off")
    k3.metric("Pending Scoring", f"{kpis['pending']:,}", delta="awaiting ML engine", delta_color="off")
    k4.metric("Avg SHI", f"{kpis['avg_shi']:.4f}", delta="healthy" if kpis['avg_shi'] >= 0.7 else "warning", delta_color="normal" if kpis['avg_shi'] >= 0.7 else "inverse")

    st.markdown("---")

    if len(df) == 0:
        st.info("⏳ Waiting for scored data… Run `python3 bridge_ml_engine.py`.")
        return

    # ── Row 1: Strain Trend + SHI Gauge ───────────────────
    c1, c2 = st.columns([5, 2])

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">📈</span>Strain Over Time (µε)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_strain(df), width="stretch", key="strain_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">🏗️</span>Structural Health Index</div>', unsafe_allow_html=True)
        gauge_fig, status_text = chart_shi_gauge(kpis["avg_shi"])
        st.plotly_chart(gauge_fig, width="stretch", key="shi_gauge")
        # Status pill
        color = C["success"] if status_text == "Healthy" else (C["warning"] if status_text == "Warning" else C["danger"])
        st.markdown(f'<div style="text-align:center;margin-top:-8px;"><span style="background:rgba({",".join(str(int(color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15);color:{color};padding:4px 16px;border-radius:12px;font-size:0.78rem;font-weight:600;">{status_text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Correlation + Anomaly Scores ───────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">🔗</span>Vehicle Load vs. Deflection</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_correlation(df), width="stretch", key="corr_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">🎯</span>Anomaly Decision Scores</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_anomaly_scores(df), width="stretch", key="score_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Vibration + Tension ────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">〰️</span>Vibration (m/s²)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_vibration(df), width="stretch", key="vib_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">🔩</span>Cable Tension (kN)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_tension(df), width="stretch", key="tension_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: Recent Anomalies Table ─────────────────────
    st.markdown("---")
    st.markdown('<div class="chart-card"><div class="chart-title"><span class="icon">🚨</span>Recent Anomalies</div>', unsafe_allow_html=True)

    anom_df = fetch_anomaly_table(engine)
    if len(anom_df) > 0:
        display = anom_df.copy()
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        display = display.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "strain_microstrain": "Strain (µε)",
            "vibration_ms2": "Vibration (m/s²)",
            "cable_member_tension_kn": "Cable Tension (kN)",
            "vehicle_load_tons": "Load (tons)",
            "anomaly_score": "Score",
        })
        for col in ["Strain (µε)", "Cable Tension (kN)", "Load (tons)"]:
            display[col] = display[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        display["Vibration (m/s²)"] = display["Vibration (m/s²)"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        display["Score"] = display["Score"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        st.dataframe(display, width="stretch", hide_index=True, height=340)
    else:
        st.info("No anomalies detected yet.")

    st.markdown('</div>', unsafe_allow_html=True)


# Run
live_dashboard()

# Footer
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px;color:{C['muted']};font-size:0.72rem;">
    Dashboard by Luctfy Alkatiri Moehtar · Auto-refresh {REFRESH_INTERVAL}s
</div>
""", unsafe_allow_html=True)
