import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Recommendation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b8fa8 !important;
        font-size: 13px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #2a2d3a;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #8b8fa8 !important;
        font-size: 13px !important;
    }

    /* Section headers */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2d3a;
    }

    /* Chart card wrapper */
    .chart-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Insight box */
    .insight-box {
        background: #1e2235;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #c4c8e0;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }

    /* Divider */
    hr { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    # Cari file CSV — sesuaikan path jika perlu
    candidates = [
        Path("data/sample_data.csv"),
        Path("../data/sample_data.csv"),
        Path("CC26-PRU466-Product-Recommendation-main/data/sample_data.csv"),
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        st.error("❌ File `data/sample_data.csv` tidak ditemukan. Letakkan file CSV di folder `data/`.")
        st.stop()

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["quarter"] = df["timestamp"].dt.quarter
    return df


df = load_data()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ Dashboard Filter")
    st.markdown("---")

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider(
        "Rentang Tahun",
        min_value=year_min,
        max_value=year_max,
        value=(2007, year_max),
    )

    rating_filter = st.multiselect(
        "Filter Rating",
        options=[1.0, 2.0, 3.0, 4.0, 5.0],
        default=[1.0, 2.0, 3.0, 4.0, 5.0],
        format_func=lambda x: f"{'★' * int(x)} ({int(x)})",
    )

    top_n = st.slider("Top N Produk / User", min_value=5, max_value=30, value=15)

    st.markdown("---")
    st.markdown("### 📊 Info Dataset")
    st.markdown(f"- **Total baris:** {len(df):,}")
    st.markdown(f"- **Periode:** {df['year'].min()} – {df['year'].max()}")
    st.markdown(f"- **Unique users:** {df['user_id'].nunique():,}")
    st.markdown(f"- **Unique products:** {df['product_id'].nunique():,}")


# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df["year"].between(*year_range) &
    df["rating"].isin(rating_filter)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#6366f1",
    "secondary": "#22d3ee",
    "success":   "#4ade80",
    "warning":   "#fbbf24",
    "danger":    "#f87171",
    "purple":    "#a78bfa",
    "pink":      "#f472b6",
}
RATING_COLORS = {
    1.0: "#f87171",
    2.0: "#fb923c",
    3.0: "#fbbf24",
    4.0: "#4ade80",
    5.0: "#22d3ee",
}
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c4c8e0", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a", tickcolor="#8b8fa8"),
    yaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a", tickcolor="#8b8fa8"),
    margin=dict(l=0, r=0, t=30, b=0),
)


# ── Helper ────────────────────────────────────────────────────────────────────
def apply_theme(fig, **overrides):
    fig.update_layout(**CHART_THEME, **overrides)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛍️ Product Recommendation Analytics")
st.markdown(
    f"<span style='color:#8b8fa8;font-size:14px'>Data {year_range[0]}–{year_range[1]} "
    f"· {len(fdf):,} ulasan · Filter aktif: {len(rating_filter)} rating</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 – KPI METRICS
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_reviews   = len(fdf)
unique_users    = fdf["user_id"].nunique()
unique_products = fdf["product_id"].nunique()
avg_rating      = fdf["rating"].mean()
pct_5star       = (fdf["rating"] == 5.0).mean() * 100
pct_1star       = (fdf["rating"] == 1.0).mean() * 100

col1.metric("📝 Total Ulasan",    f"{total_reviews:,}")
col2.metric("👤 Pengguna Unik",   f"{unique_users:,}")
col3.metric("📦 Produk Unik",     f"{unique_products:,}")
col4.metric("⭐ Avg Rating",      f"{avg_rating:.2f}")
col5.metric("🟢 Rating 5★",       f"{pct_5star:.1f}%")
col6.metric("🔴 Rating 1★",       f"{pct_1star:.1f}%")

st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 – TREN TAHUNAN + DISTRIBUSI RATING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Tren & Distribusi</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([2, 1])

with col_left:
    yearly = (
        fdf.groupby("year")
        .agg(count=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=yearly["year"], y=yearly["count"],
            name="Jumlah Ulasan",
            marker_color=COLORS["primary"],
            opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["year"], y=yearly["avg_rating"].round(2),
            name="Avg Rating",
            mode="lines+markers",
            line=dict(color=COLORS["warning"], width=2.5),
            marker=dict(size=7, color=COLORS["warning"]),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Jumlah Ulasan & Rata-rata Rating per Tahun",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **CHART_THEME,
    )
    fig.update_yaxes(title_text="Jumlah Ulasan", secondary_y=False,
                     gridcolor="#2a2d3a", tickcolor="#8b8fa8")
    fig.update_yaxes(title_text="Avg Rating", secondary_y=True,
                     range=[3.5, 5.0], gridcolor="rgba(0,0,0,0)", tickcolor="#8b8fa8")
    fig.update_xaxes(gridcolor="#2a2d3a")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    dist = fdf["rating"].value_counts().sort_index().reset_index()
    dist.columns = ["rating", "count"]
    dist["label"] = dist["rating"].map(lambda x: f"{'★'*int(x)} ({int(x)})")
    dist["color"] = dist["rating"].map(RATING_COLORS)
    dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(1)

    fig2 = go.Figure(go.Pie(
        labels=dist["label"],
        values=dist["count"],
        hole=0.6,
        marker=dict(colors=dist["color"], line=dict(color="#0f1117", width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="#fff"),
        hovertemplate="<b>%{label}</b><br>%{value:,} ulasan (%{percent})<extra></extra>",
    ))
    fig2.add_annotation(
        text=f"<b>{avg_rating:.2f}</b><br><span style='font-size:11px'>avg ★</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e2e8f0"),
        align="center",
    )
    fig2.update_layout(
    title="Distribusi Rating",
    showlegend=True,
    legend=dict(orientation="v", font=dict(size=11)),
    **CHART_THEME,
    margin=dict(l=0, r=0, t=30, b=0),  # ← HAPUS baris ini
)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 – TREN BULANAN
# ═══════════════════════════════════════════════════════════════════════════════
monthly = (
    fdf.groupby("year_month")
    .agg(count=("rating", "count"), avg_rating=("rating", "mean"))
    .reset_index()
    .sort_values("year_month")
)

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(
    go.Scatter(
        x=monthly["year_month"], y=monthly["count"],
        name="Jumlah Ulasan",
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color=COLORS["primary"], width=1.5),
        mode="lines",
    ),
    secondary_y=False,
)
fig3.add_trace(
    go.Scatter(
        x=monthly["year_month"], y=monthly["avg_rating"].round(2),
        name="Avg Rating",
        line=dict(color=COLORS["warning"], width=2, dash="dot"),
        mode="lines",
    ),
    secondary_y=True,
)
fig3.update_layout(
    title="Tren Bulanan: Volume Ulasan & Rating",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    **CHART_THEME,
)
fig3.update_yaxes(title_text="Jumlah Ulasan", secondary_y=False, gridcolor="#2a2d3a")
fig3.update_yaxes(title_text="Avg Rating", secondary_y=True,
                  range=[3.5, 5.0], gridcolor="rgba(0,0,0,0)")
fig3.update_xaxes(nticks=20, tickangle=45, gridcolor="#2a2d3a")
st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 – HEATMAP RATING (YEAR × MONTH)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗓️ Heatmap & Pola Musiman</div>', unsafe_allow_html=True)

col_heat, col_box = st.columns([3, 2])

with col_heat:
    heat = (
        fdf.groupby(["year", "month"])["rating"]
        .mean()
        .reset_index()
        .pivot(index="year", columns="month", values="rating")
    )
    month_labels = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    cols_present = [c for c in range(1, 13) if c in heat.columns]
    heat = heat[cols_present]

    fig4 = go.Figure(go.Heatmap(
        z=heat.values.round(2),
        x=[month_labels[c-1] for c in cols_present],
        y=heat.index.astype(str),
        colorscale=[
            [0.0, "#f87171"], [0.25, "#fb923c"],
            [0.5,  "#fbbf24"], [0.75, "#4ade80"],
            [1.0,  "#22d3ee"],
        ],
        zmin=3.5, zmax=5.0,
        text=heat.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Tahun %{y} %{x}<br>Avg rating: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Rating", tickfont=dict(color="#c4c8e0")),
    ))
    fig4.update_layout(title="Heatmap Avg Rating (Tahun × Bulan)", **CHART_THEME)
    st.plotly_chart(fig4, use_container_width=True)

with col_box:
    fig5 = go.Figure()
    for yr in sorted(fdf["year"].unique()):
        subset = fdf[fdf["year"] == yr]["rating"]
        fig5.add_trace(go.Box(
            y=subset, name=str(yr),
            marker_color=COLORS["primary"],
            line_color=COLORS["secondary"],
            boxmean=True,
        ))
    fig5.update_layout(
        title="Distribusi Rating per Tahun",
        showlegend=False,
        **CHART_THEME,
        yaxis=dict(range=[0.5, 5.5], gridcolor="#2a2d3a"),
        xaxis=dict(gridcolor="#2a2d3a", tickangle=45),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5 – TOP PRODUK & TOP USER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🏆 Top Produk & Pengguna</div>', unsafe_allow_html=True)
col_prod, col_user = st.columns(2)

with col_prod:
    tab1, tab2 = st.tabs(["📈 Terlaris (volume)", "⭐ Tertinggi (rating)"])

    with tab1:
        top_vol = (
            fdf.groupby("product_id")
            .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"))
            .sort_values("reviews", ascending=False)
            .head(top_n)
            .reset_index()
        )
        top_vol["avg_rating"] = top_vol["avg_rating"].round(2)
        fig6 = go.Figure(go.Bar(
            x=top_vol["reviews"],
            y=top_vol["product_id"],
            orientation="h",
            marker=dict(
                color=top_vol["avg_rating"],
                colorscale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#22d3ee"]],
                colorbar=dict(title="Avg ★", len=0.6, thickness=12),
                cmin=1, cmax=5,
            ),
            text=top_vol["reviews"].map(lambda x: f"{x:,}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Ulasan: %{x:,}<br>Avg rating: %{marker.color:.2f}<extra></extra>",
        ))
        fig6.update_layout(
            title=f"Top {top_n} Produk berdasarkan Volume",
            height=max(350, top_n * 28),
            **CHART_THEME,
            xaxis=dict(gridcolor="#2a2d3a"),
            yaxis=dict(autorange="reversed", gridcolor="#2a2d3a"),
        )
        st.plotly_chart(fig6, use_container_width=True)

    with tab2:
        top_rat = (
            fdf.groupby("product_id")
            .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"))
            .query("reviews >= 10")
            .sort_values("avg_rating", ascending=False)
            .head(top_n)
            .reset_index()
        )
        top_rat["avg_rating"] = top_rat["avg_rating"].round(2)
        fig7 = go.Figure(go.Bar(
            x=top_rat["avg_rating"],
            y=top_rat["product_id"],
            orientation="h",
            marker=dict(
                color=top_rat["reviews"],
                colorscale=[[0, COLORS["primary"]], [1, COLORS["secondary"]]],
                colorbar=dict(title="Ulasan", len=0.6, thickness=12),
            ),
            text=top_rat["avg_rating"].map(lambda x: f"★ {x:.2f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg rating: %{x:.2f}<br>Ulasan: %{marker.color:,}<extra></extra>",
        ))
        fig7.update_layout(
            title=f"Top {top_n} Produk berdasarkan Rating (min. 10 ulasan)",
            height=max(350, top_n * 28),
            **CHART_THEME,
            xaxis=dict(range=[4.0, 5.2], gridcolor="#2a2d3a"),
            yaxis=dict(autorange="reversed", gridcolor="#2a2d3a"),
        )
        st.plotly_chart(fig7, use_container_width=True)

with col_user:
    user_stats = (
        fdf.groupby("user_id")
        .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"))
        .sort_values("reviews", ascending=False)
        .head(top_n)
        .reset_index()
    )
    user_stats["avg_rating"] = user_stats["avg_rating"].round(2)

    fig8 = go.Figure(go.Bar(
        x=user_stats["reviews"],
        y=user_stats["user_id"],
        orientation="h",
        marker=dict(
            color=user_stats["avg_rating"],
            colorscale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#22d3ee"]],
            colorbar=dict(title="Avg ★", len=0.6, thickness=12),
            cmin=1, cmax=5,
        ),
        text=user_stats["reviews"].map(lambda x: f"{x:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Ulasan: %{x:,}<br>Avg rating: %{marker.color:.2f}<extra></extra>",
    ))
    fig8.update_layout(
        title=f"Top {top_n} Reviewer Paling Aktif",
        height=max(350, top_n * 28),
        **CHART_THEME,
        xaxis=dict(gridcolor="#2a2d3a"),
        yaxis=dict(autorange="reversed", gridcolor="#2a2d3a"),
    )
    st.plotly_chart(fig8, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6 – SEGMENTASI PENGGUNA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👥 Segmentasi Pengguna</div>', unsafe_allow_html=True)

all_users = (
    fdf.groupby("user_id")
    .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"))
    .reset_index()
)
bins   = [0, 1, 5, 20, 50, 10000]
labels = ["Casual (1)", "Light (2-5)", "Moderate (6-20)", "Active (21-50)", "Power (50+)"]
all_users["segment"] = pd.cut(all_users["reviews"], bins=bins, labels=labels)

col_seg1, col_seg2, col_seg3 = st.columns(3)

with col_seg1:
    seg_count = all_users["segment"].value_counts().reset_index()
    seg_count.columns = ["segment", "count"]
    seg_colors = [COLORS["danger"], COLORS["warning"], COLORS["primary"],
                  COLORS["secondary"], COLORS["success"]]
    fig9 = go.Figure(go.Pie(
        labels=seg_count["segment"],
        values=seg_count["count"],
        hole=0.5,
        marker=dict(colors=seg_colors, line=dict(color="#0f1117", width=2)),
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig9.update_layout(title="Distribusi Segmen Pengguna",
                       showlegend=False, **CHART_THEME,
                       margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig9, use_container_width=True)

with col_seg2:
    seg_avg = all_users.groupby("segment", observed=True)["avg_rating"].mean().reset_index()
    fig10 = go.Figure(go.Bar(
        x=seg_avg["segment"],
        y=seg_avg["avg_rating"].round(2),
        marker_color=seg_colors,
        text=seg_avg["avg_rating"].round(2),
        textposition="outside",
    ))
    fig10.update_layout(
        title="Avg Rating per Segmen",
        yaxis=dict(range=[3.5, 5.0], gridcolor="#2a2d3a"),
        xaxis=dict(gridcolor="#2a2d3a", tickangle=15),
        **CHART_THEME,
    )
    st.plotly_chart(fig10, use_container_width=True)

with col_seg3:
    seg_vol = all_users.groupby("segment", observed=True)["reviews"].sum().reset_index()
    fig11 = go.Figure(go.Bar(
        x=seg_vol["segment"],
        y=seg_vol["reviews"],
        marker_color=seg_colors,
        text=seg_vol["reviews"].map(lambda x: f"{x:,}"),
        textposition="outside",
    ))
    fig11.update_layout(
        title="Total Ulasan per Segmen",
        yaxis=dict(gridcolor="#2a2d3a"),
        xaxis=dict(gridcolor="#2a2d3a", tickangle=15),
        **CHART_THEME,
    )
    st.plotly_chart(fig11, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 7 – SCATTER: REVIEW COUNT vs AVG RATING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔬 Analisis Lanjutan</div>', unsafe_allow_html=True)

col_sc, col_quarter = st.columns(2)

with col_sc:
    prod_scatter = (
        fdf.groupby("product_id")
        .agg(reviews=("rating", "count"), avg_rating=("rating", "mean"))
        .query("reviews >= 5")
        .reset_index()
    )
    fig12 = go.Figure(go.Scatter(
        x=prod_scatter["reviews"],
        y=prod_scatter["avg_rating"].round(2),
        mode="markers",
        marker=dict(
            size=6,
            color=prod_scatter["avg_rating"],
            colorscale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#22d3ee"]],
            opacity=0.6,
            colorbar=dict(title="Rating", len=0.7, thickness=12),
        ),
        hovertemplate="<b>%{text}</b><br>Ulasan: %{x:,}<br>Avg rating: %{y:.2f}<extra></extra>",
        text=prod_scatter["product_id"],
    ))
    fig12.update_layout(
        title="Produk: Volume Ulasan vs Avg Rating (min. 5 ulasan)",
        xaxis=dict(title="Jumlah Ulasan", type="log", gridcolor="#2a2d3a"),
        yaxis=dict(title="Avg Rating", range=[0.5, 5.5], gridcolor="#2a2d3a"),
        **CHART_THEME,
    )
    st.plotly_chart(fig12, use_container_width=True)

with col_quarter:
    qtr = (
        fdf.groupby(["year", "quarter"])
        .agg(count=("rating", "count"), avg=("rating", "mean"))
        .reset_index()
    )
    qtr["label"] = qtr["year"].astype(str) + " Q" + qtr["quarter"].astype(str)
    fig13 = go.Figure()
    for q in [1, 2, 3, 4]:
        sub = qtr[qtr["quarter"] == q]
        fig13.add_trace(go.Scatter(
            x=sub["label"], y=sub["avg"].round(2),
            name=f"Q{q}",
            mode="lines+markers",
            marker=dict(size=6),
        ))
    fig13.update_layout(
        title="Tren Avg Rating per Kuartal",
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(tickangle=45, gridcolor="#2a2d3a"),
        yaxis=dict(range=[3.5, 5.0], gridcolor="#2a2d3a"),
        **CHART_THEME,
    )
    st.plotly_chart(fig13, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 8 – DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Tabel Data Mentah</div>', unsafe_allow_html=True)

with st.expander("Lihat data mentah (100 baris pertama)", expanded=False):
    show_df = fdf[["user_id", "product_id", "rating", "timestamp"]].head(100).copy()
    show_df["timestamp"] = show_df["timestamp"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        show_df,
        use_container_width=True,
        column_config={
            "user_id":    st.column_config.TextColumn("User ID"),
            "product_id": st.column_config.TextColumn("Product ID"),
            "rating":     st.column_config.NumberColumn("Rating", format="%.1f ★"),
            "timestamp":  st.column_config.TextColumn("Tanggal"),
        },
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a4d5e;font-size:12px'>"
    "Product Recommendation Dashboard · CC26-PRU466 · Data: sample_data.csv"
    "</div>",
    unsafe_allow_html=True,
)
