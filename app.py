from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Data Insights | Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
NAVY = "#172554"
BLUE = "#2563EB"
TEAL = "#0F766E"
MUTED = "#64748B"
PALETTE = ["#2563EB", "#0F766E", "#F59E0B", "#7C3AED", "#DB2777", "#0891B2"]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F6F8FC; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E7ECF4; }
    .hero { padding: 1.15rem 1.35rem; border-radius: 18px;
            background: linear-gradient(115deg, #172554 0%, #1D4ED8 62%, #0F766E 120%);
            color: white; margin-bottom: 1rem; }
    .hero h1 { color: white; margin: 0; font-size: 2rem; letter-spacing: -0.04em; }
    .hero p { color: #DBEAFE; margin: .35rem 0 0 0; }
    .kpi { background: white; border: 1px solid #E7ECF4; border-radius: 15px;
           padding: 1rem 1.05rem; min-height: 112px; box-shadow: 0 3px 12px rgba(15,23,42,.035); }
    .kpi-label { color: #64748B; font-size: .83rem; font-weight: 600; }
    .kpi-value { color: #172554; font-size: 1.75rem; font-weight: 700; margin-top: .25rem; }
    .kpi-note { color: #94A3B8; font-size: .75rem; margin-top: .15rem; }
    .section-title { color: #172554; font-size: 1.2rem; font-weight: 700; margin: .6rem 0 .1rem 0; }
    .section-sub { color: #64748B; font-size: .88rem; margin-bottom: .8rem; }
    div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E7ECF4;
        border-radius: 14px; padding: .35rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #E7ECF4; border-radius: 12px; overflow: hidden; }
    .insight { padding: .85rem 1rem; background: #EFF6FF; border-left: 4px solid #2563EB;
        border-radius: 8px; color: #1E3A8A; margin: .35rem 0; }
    .small-muted { color: #64748B; font-size: .8rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Data loading ----------
DEFAULT_DRIVE_ID = "1lHdwPmg9bcliWzxHb6UZmNK8ARaL5VFE"
DEFAULT_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DEFAULT_DRIVE_ID}"

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    """Load CSV or Excel from a direct URL. Google Drive files must be publicly accessible."""
    if not url.strip():
        raise ValueError("Please enter a dataset URL.")
    if "drive.google.com/file/d/" in url:
        match = re.search(r"/file/d/([^/]+)", url)
        if match:
            url = f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    elif "drive.google.com" in url and "id=" not in url:
        raise ValueError("Could not identify a Google Drive file ID in this URL.")
    if "drive.google.com" in url:
        try:
            frame = pd.read_csv(url)
            if frame.shape[1] == 1 and str(frame.columns[0]).lower().startswith("<!doctype"):
                raise ValueError("Google Drive returned a web page instead of a dataset.")
            return frame
        except Exception:
            try:
                return pd.read_excel(url)
            except Exception as exc:
                raise ValueError(
                    "Could not read the Drive file. Check that General access is set to "
                    "'Anyone with the link' and that the file is CSV or Excel."
                ) from exc
    if url.lower().split("?")[0].endswith((".xlsx", ".xls")):
        return pd.read_excel(url)
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    if suffix in (".csv", ".txt"):
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    if suffix == ".json":
        return pd.read_json(io.BytesIO(file_bytes))
    raise ValueError("Supported formats: CSV, XLSX, XLS, JSON, TXT.")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(c).strip())).strip("_") or f"column_{i+1}"
        for i, c in enumerate(out.columns)
    ]
    # Make duplicate normalized names unique.
    seen, cols = {}, []
    for c in out.columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    out.columns = cols
    return out

def infer_dates_and_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            series = out[col].astype("string").str.strip()
            # Parse likely date columns conservatively.
            if re.search(r"date|time|day|month|year", col, re.I):
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().mean() >= 0.65:
                    out[col] = parsed
                    continue
            # Numeric conversion only when most non-empty values parse and the column is not ID-like.
            if not re.search(r"(^id$|_id$|id_|code|zip|postal|phone|sku)", col, re.I):
                cleaned = series.str.replace(r"[$€£₹,%]", "", regex=True).str.replace(",", "", regex=False)
                nums = pd.to_numeric(cleaned, errors="coerce")
                nonempty = series.notna() & (series != "")
                if nonempty.sum() and nums[nonempty].notna().mean() >= 0.85:
                    out[col] = nums
    return out

def format_value(value, col_name="", series=None):
    name = col_name.lower()
    if pd.isna(value):
        return "—"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%d %b %Y")
    if isinstance(value, (int, np.integer)):
        return f"{value:,.0f}"
    if isinstance(value, (float, np.floating)):
        if any(k in name for k in ("margin", "rate", "percent", "percentage", "conversion")) and abs(value) <= 1:
            return f"{value:.1%}"
        if any(k in name for k in ("revenue", "sales", "price", "profit", "cost", "amount", "spend", "salary", "value")):
            return f"{value:,.2f}"
        return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)

def make_chart(fig, height=370):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=18, r=18, t=48, b=18),
        font=dict(family="DM Sans, sans-serif", color="#334155", size=12),
        title_font=dict(size=15, color=NAVY), legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", automargin=True)
    fig.update_yaxes(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0", automargin=True)
    return fig

def section(title, subtitle):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def select_columns(df, predicate, fallback=True):
    cols = [c for c in df.columns if predicate(c, df[c])]
    return cols if cols else (list(df.columns) if fallback else [])

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📊 Data Insights")
    st.caption("Interactive analytics workspace")
    source_mode = st.radio("Data source", ["Google Drive URL", "Upload a file"], index=0)
    uploaded = None
    drive_url = DEFAULT_DRIVE_URL
    if source_mode == "Google Drive URL":
        drive_url = st.text_input("Public dataset URL", value=DEFAULT_DRIVE_URL,
                                  help="The file must be accessible to anyone with the link.")
        st.caption("Tip: Drive → Share → General access → Anyone with the link.")
    else:
        uploaded = st.file_uploader("Choose dataset", type=["csv", "xlsx", "xls", "json", "txt"])
    st.divider()
    st.markdown("**Dashboard controls**")
    st.caption("Filters are generated from the detected columns after the data loads.")

try:
    with st.spinner("Loading and profiling dataset…"):
        if source_mode == "Upload a file":
            if uploaded is None:
                st.markdown('<div class="hero"><h1>Data Insights Dashboard</h1><p>Upload your dataset to explore performance, trends, segments, and data quality.</p></div>', unsafe_allow_html=True)
                st.info("Upload the dataset from the sidebar to start. CSV, Excel, JSON, and TXT files are supported.")
                st.stop()
            raw_df = load_uploaded_file(uploaded.getvalue(), uploaded.name)
        else:
            raw_df = load_from_url(drive_url)
except Exception as exc:
    st.markdown('<div class="hero"><h1>Data Insights Dashboard</h1><p>Explore your dataset through an interactive analytics workspace.</p></div>', unsafe_allow_html=True)
    st.error(f"Dataset could not be loaded: {exc}")
    st.info("If this is a Google Drive file, make sure it is shared as 'Anyone with the link'. You can also switch to 'Upload a file' and upload the downloaded dataset.")
    st.stop()

if raw_df.empty:
    st.error("The dataset loaded, but it contains no rows.")
    st.stop()

original_rows, original_cols = raw_df.shape
df = clean_column_names(raw_df)
df = infer_dates_and_numbers(df)
duplicate_count = int(df.duplicated().sum())
missing_cells = int(df.isna().sum().sum())
missing_pct = (missing_cells / max(df.shape[0] * df.shape[1], 1)) * 100

date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in date_cols and c not in num_cols and df[c].nunique(dropna=True) <= max(100, min(500, int(len(df)*0.5)))]
cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) > 1]
id_cols = [c for c in df.columns if re.search(r"(^id$|_id$|id_|code$|^index$)", c, re.I)]
measure_cols = [c for c in num_cols if c not in id_cols and df[c].nunique(dropna=True) > 1]
if not measure_cols:
    measure_cols = num_cols[:]

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <h1>Data Insights Dashboard</h1>
  <p>Explore patterns, compare segments, and uncover opportunities in your data.</p>
</div>
""", unsafe_allow_html=True)
meta1, meta2, meta3 = st.columns([1, 1, 2])
meta1.caption(f"**{len(df):,}** records")
meta2.caption(f"**{len(df.columns)}** fields")
meta3.caption("Interactive analytics · Data quality checks · Exportable records")

# ---------- Filters ----------
with st.expander("🔎 Explore & filter data", expanded=True):
    filter_cols = st.columns(3)
    filtered = df.copy()
    active_filter_count = 0
    for i, col in enumerate(cat_cols[:6]):
        values = sorted(df[col].dropna().astype(str).unique().tolist())
        if 1 < len(values) <= 100:
            with filter_cols[i % 3]:
                chosen = st.multiselect(f"{col.replace('_',' ').title()}", values, default=[], key=f"filter_{col}")
            if chosen:
                filtered = filtered[filtered[col].astype(str).isin(chosen)]
                active_filter_count += 1
    if date_cols:
        with filter_cols[0]:
            dcol = st.selectbox("Date field", date_cols, key="date_filter_col")
            valid_dates = filtered[dcol].dropna()
            if not valid_dates.empty:
                min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
                date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="date_range")
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filtered = filtered[filtered[dcol].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))]
                    active_filter_count += 1
    if measure_cols:
        with filter_cols[1]:
            metric_filter = st.selectbox("Numeric range filter (optional)", ["No numeric filter"] + measure_cols, key="numeric_filter_col")
        if metric_filter != "No numeric filter":
            vals = filtered[metric_filter].dropna()
            if not vals.empty:
                lo, hi = float(vals.min()), float(vals.max())
                if lo < hi:
                    with filter_cols[2]:
                        bounds = st.slider(f"Range: {metric_filter}", min_value=lo, max_value=hi, value=(lo, hi), key="numeric_bounds")
                    filtered = filtered[filtered[metric_filter].between(bounds[0], bounds[1])]
                    active_filter_count += 1
    st.caption(f"Showing **{len(filtered):,} of {len(df):,}** records · {active_filter_count} active filter(s)")

if filtered.empty:
    st.warning("No records match the selected filters. Adjust or clear one or more filters.")
    st.stop()

# ---------- KPI cards ----------
section("At a glance", "Core dataset metrics update automatically when you change filters.")
kpis = st.columns(4)
kpis[0].markdown(f'<div class="kpi"><div class="kpi-label">Filtered records</div><div class="kpi-value">{len(filtered):,}</div><div class="kpi-note">of {len(df):,} total records</div></div>', unsafe_allow_html=True)
kpis[1].markdown(f'<div class="kpi"><div class="kpi-label">Fields</div><div class="kpi-value">{len(df.columns):,}</div><div class="kpi-note">columns available for analysis</div></div>', unsafe_allow_html=True)
kpis[2].markdown(f'<div class="kpi"><div class="kpi-label">Missing cells</div><div class="kpi-value">{int(filtered.isna().sum().sum()):,}</div><div class="kpi-note">{(filtered.isna().sum().sum()/max(filtered.size,1))*100:.1f}% of visible cells</div></div>', unsafe_allow_html=True)
if measure_cols:
    default_metric = next((c for c in measure_cols if re.search(r"sales|revenue|profit|amount|total|price|value", c, re.I)), measure_cols[0])
    metric = st.selectbox("Primary metric", measure_cols, index=measure_cols.index(default_metric), key="primary_metric")
    metric_sum = filtered[metric].sum(min_count=1)
    metric_label = metric.replace("_", " ").title()
    kpis[3].markdown(f'<div class="kpi"><div class="kpi-label">Total {metric_label}</div><div class="kpi-value">{format_value(metric_sum, metric)}</div><div class="kpi-note">sum of selected metric</div></div>', unsafe_allow_html=True)
else:
    metric = None
    kpis[3].markdown(f'<div class="kpi"><div class="kpi-label">Duplicate rows</div><div class="kpi-value">{duplicate_count:,}</div><div class="kpi-note">exact duplicate records in source</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Trends ----------
section("Trend analysis", "Understand how the selected metric changes over time.")
if date_cols and measure_cols:
    trend_cols = st.columns([1, 1, 2])
    with trend_cols[0]:
        trend_date = st.selectbox("Trend date", date_cols, key="trend_date")
    with trend_cols[1]:
        aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count"], key="trend_agg")
    with trend_cols[2]:
        grain = st.selectbox("Time grain", ["Auto", "Day", "Week", "Month", "Quarter", "Year"], index=2, key="trend_grain")
    temp = filtered[[trend_date] + ([metric] if metric else [])].dropna(subset=[trend_date]).copy()
    if not temp.empty:
        if grain == "Auto":
            span = (temp[trend_date].max() - temp[trend_date].min()).days
            grain = "Day" if span <= 45 else ("Week" if span <= 180 else "Month")
        freq = {"Day":"D", "Week":"W", "Month":"MS", "Quarter":"QS", "Year":"YS"}[grain]
        temp["_period"] = temp[trend_date].dt.to_period({"D":"D","W":"W","MS":"M","QS":"Q","YS":"Y"}[freq]).dt.start_time
        if aggregation == "Count" or metric is None:
            trend = temp.groupby("_period").size().rename("Records").reset_index()
            ycol = "Records"
        else:
            fn = {"Sum":"sum", "Mean":"mean", "Median":"median"}[aggregation]
            trend = temp.groupby("_period")[metric].agg(fn).reset_index()
            ycol = metric
        fig = px.line(trend, x="_period", y=ycol, markers=True, title=f"{aggregation} by {grain.lower()}",
                      labels={"_period":"Period", ycol: ycol.replace("_"," ").title()},
                      color_discrete_sequence=[BLUE], template="plotly_white")
        fig.update_traces(line=dict(width=3), marker=dict(size=6), hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>")
        st.plotly_chart(make_chart(fig, 390), use_container_width=True)
    else:
        st.info("No valid dates are available for the selected filters.")
else:
    st.info("A date field and numeric measure are needed for time-series analysis. The other sections remain available.")

# ---------- Category analysis ----------
section("Segment performance", "Compare groups to identify the largest contributors and meaningful differences.")
if cat_cols and measure_cols:
    left, right = st.columns(2)
    with left:
        category = st.selectbox("Segment field", cat_cols, key="segment_category")
        top_n = st.slider("Number of categories", 5, 25, 10, key="segment_topn")
    with right:
        segment_metric = st.selectbox("Measure", measure_cols, index=measure_cols.index(metric) if metric in measure_cols else 0, key="segment_metric")
        segment_agg = st.selectbox("Measure aggregation", ["Sum", "Mean", "Median", "Count"], key="segment_agg")
    if segment_agg == "Count":
        grouped = filtered.groupby(category, dropna=False).size().rename("Records").reset_index()
        val_col = "Records"
    else:
        fn = {"Sum":"sum", "Mean":"mean", "Median":"median"}[segment_agg]
        grouped = filtered.groupby(category, dropna=False)[segment_metric].agg(fn).reset_index()
        val_col = segment_metric
    grouped[category] = grouped[category].astype("string").fillna("(Missing)")
    grouped = grouped.sort_values(val_col, ascending=False).head(top_n)
    fig = px.bar(grouped.sort_values(val_col), x=val_col, y=category, orientation="h",
                 title=f"{segment_agg} of {val_col.replace('_',' ').title()} by {category.replace('_',' ').title()}",
                 labels={val_col: val_col.replace("_"," ").title(), category: category.replace("_"," ").title()},
                 color=val_col, color_continuous_scale=["#BFDBFE", BLUE], text_auto=".3s")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(make_chart(fig, max(350, min(650, 30*len(grouped)+100))), use_container_width=True)
    if len(cat_cols) > 1:
        c2, c3 = st.columns(2)
        with c2:
            second_cat = st.selectbox("Secondary segment", [c for c in cat_cols if c != category], key="secondary_category")
            cross = pd.crosstab(filtered[category].fillna("(Missing)").astype(str),
                                filtered[second_cat].fillna("(Missing)").astype(str))
            if cross.shape[0] > 1 and cross.shape[1] > 1:
                fig = px.imshow(cross, aspect="auto", color_continuous_scale="Blues",
                                title=f"Record volume: {category} × {second_cat}",
                                labels=dict(x=second_cat.replace("_"," ").title(), y=category.replace("_"," ").title(), color="Records"))
                st.plotly_chart(make_chart(fig, 360), use_container_width=True)
            else:
                st.info("Not enough segment variety for a cross-tab heatmap.")
        with c3:
            share = grouped.copy()
            share["_share"] = share[val_col] / grouped[val_col].sum() * 100 if grouped[val_col].sum() else 0
            fig = px.pie(share, names=category, values=val_col, hole=.62,
                         title=f"Share across top {len(share)} segments",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value:,.2f}<br>%{percent}<extra></extra>")
            st.plotly_chart(make_chart(fig, 360), use_container_width=True)
else:
    st.info("Categorical and numeric fields are needed for segment comparisons.")

# ---------- Distribution and relationships ----------
section("Distribution & relationships", "Inspect spread, skew, unusual values, and relationships between measures.")
if measure_cols:
    dleft, dright = st.columns(2)
    with dleft:
        dist_metric = st.selectbox("Distribution measure", measure_cols, index=measure_cols.index(metric) if metric in measure_cols else 0, key="dist_metric")
        vals = filtered[dist_metric].dropna()
        fig = px.histogram(filtered, x=dist_metric, nbins=35, marginal="box",
                           title=f"Distribution of {dist_metric.replace('_',' ').title()}",
                           color_discrete_sequence=[TEAL], opacity=.88)
        fig.update_traces(hovertemplate=f"{dist_metric}: "+"%{x:,.3g}<br>Records: %{y}<extra></extra>")
        st.plotly_chart(make_chart(fig, 380), use_container_width=True)
        if len(vals) > 0:
            st.caption(f"Mean: **{vals.mean():,.3g}** · Median: **{vals.median():,.3g}** · Std. dev.: **{vals.std():,.3g}**")
    with dright:
        if len(measure_cols) >= 2:
            x_metric = st.selectbox("X-axis measure", measure_cols, key="scatter_x")
            y_options = [c for c in measure_cols if c != x_metric] or measure_cols
            y_metric = st.selectbox("Y-axis measure", y_options, key="scatter_y")
            color_options = ["No color grouping"] + cat_cols
            color_by = st.selectbox("Color by (optional)", color_options, key="scatter_color")
            plot_df = filtered[[x_metric, y_metric] + ([color_by] if color_by != "No color grouping" else [])].dropna()
            if len(plot_df) > 5000:
                plot_df = plot_df.sample(5000, random_state=42)
            fig = px.scatter(plot_df, x=x_metric, y=y_metric,
                             color=None if color_by == "No color grouping" else color_by,
                             opacity=.72, trendline="ols" if len(plot_df) > 2 else None,
                             title=f"{y_metric.replace('_',' ').title()} vs {x_metric.replace('_',' ').title()}",
                             color_discrete_sequence=PALETTE)
            st.plotly_chart(make_chart(fig, 380), use_container_width=True)
            corr = filtered[[x_metric, y_metric]].corr(numeric_only=True).iloc[0,1]
            st.caption(f"Pearson correlation: **{corr:.3f}** · Correlation is descriptive and does not imply causation.")
        else:
            st.info("At least two numeric measures are needed for relationship analysis.")
else:
    st.info("No numeric measures were detected for distribution analysis.")

# ---------- Business insights ----------
section("Automated observations", "Data-led prompts to help focus further investigation. Validate business context before acting.")
insights = []
insights.append(f"The current view contains {len(filtered):,} records ({len(filtered)/max(len(df),1):.1%} of the source dataset).")
if filtered.isna().sum().sum():
    worst_col = filtered.isna().mean().sort_values(ascending=False).index[0]
    insights.append(f"'{worst_col}' has the highest missing-value rate in the filtered view ({filtered[worst_col].isna().mean():.1%}); investigate whether this is expected.")
else:
    insights.append("No missing cells remain in the current filtered view.")
if duplicate_count:
    insights.append(f"The original data contains {duplicate_count:,} exact duplicate rows. Confirm whether they are valid repeated events before removing them.")
if measure_cols and cat_cols:
    best_cat = cat_cols[0]
    best_measure = metric if metric else measure_cols[0]
    rank = filtered.groupby(best_cat, dropna=False)[best_measure].sum(min_count=1).sort_values(ascending=False)
    if not rank.empty:
        insights.append(f"'{rank.index[0]}' leads on total {best_measure.replace('_',' ')} ({format_value(rank.iloc[0], best_measure)}). Compare its volume and mix with other segments.")
if measure_cols:
    chosen_vals = filtered[metric].dropna() if metric else filtered[measure_cols[0]].dropna()
    if len(chosen_vals) >= 5:
        q1, q3 = chosen_vals.quantile([.25, .75])
        iqr = q3-q1
        outliers = int(((chosen_vals < q1-1.5*iqr) | (chosen_vals > q3+1.5*iqr)).sum()) if iqr > 0 else 0
        insights.append(f"{outliers:,} potential IQR outlier(s) detected in '{metric or measure_cols[0]}'. Review them in context; unusual values may be legitimate.")
for item in insights:
    st.markdown(f'<div class="insight">↳ {item}</div>', unsafe_allow_html=True)

# ---------- Data quality ----------
section("Data quality & profiling", "Review the cleaning checks before using the results in reporting.")
quality = pd.DataFrame({
    "Column": df.columns,
    "Data type": [str(df[c].dtype) for c in df.columns],
    "Non-null values": [int(df[c].notna().sum()) for c in df.columns],
    "Missing values": [int(df[c].isna().sum()) for c in df.columns],
    "Missing %": [round(df[c].isna().mean()*100, 2) for c in df.columns],
    "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
q1, q2, q3, q4 = st.columns(4)
q1.metric("Source rows", f"{original_rows:,}")
q2.metric("Source columns", f"{original_cols:,}")
q3.metric("Exact duplicates", f"{duplicate_count:,}")
q4.metric("Missing cells", f"{missing_cells:,}", f"{missing_pct:.1f}%")
st.dataframe(quality, use_container_width=True, hide_index=True)
with st.expander("Statistical summary"):
    st.dataframe(filtered.describe(include="all").transpose().reset_index(names="Field"), use_container_width=True)

# Optional Seaborn visual: correlation heatmap adds value when multiple numeric measures exist.
if len(measure_cols) >= 2:
    with st.expander("Statistical view: correlation heatmap (Seaborn)"):
        corr_data = filtered[measure_cols].corr(numeric_only=True)
        fig, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(9, 5))
        sns.heatmap(corr_data, cmap="vlag", center=0, annot=len(measure_cols) <= 8,
                    fmt=".2f", linewidths=.4, ax=ax, cbar_kws={"shrink": .8})
        ax.set_title("Correlation between numeric fields", pad=14, weight="bold")
        st.pyplot(fig, clear_figure=True)
        st.caption("Correlation measures linear association; it does not establish causality.")

# ---------- Detail table / downloads ----------
section("Detailed records", "Search, sort, and download the records in the current filtered view.")
search = st.text_input("Search visible records", placeholder="Type a value to search across columns…", key="table_search")
display_df = filtered
if search.strip():
    mask = display_df.astype(str).apply(lambda col: col.str.contains(re.escape(search.strip()), case=False, na=False)).any(axis=1)
    display_df = display_df[mask]
st.caption(f"{len(display_df):,} matching records")
st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)
download_col1, download_col2 = st.columns(2)
download_col1.download_button("⬇ Download filtered data (CSV)", data=display_df.to_csv(index=False).encode("utf-8"),
                              file_name="filtered_analytics_data.csv", mime="text/csv", use_container_width=True)
download_col2.download_button("⬇ Download data quality report (CSV)", data=quality.to_csv(index=False).encode("utf-8"),
                              file_name="data_quality_report.csv", mime="text/csv", use_container_width=True)

st.divider()
st.markdown('<p class="small-muted">Data Insights Dashboard · Built with Streamlit, Pandas, Plotly, and Seaborn. Metrics and observations are generated from the selected data view.</p>', unsafe_allow_html=True)
