"""
FLUXORA — AI-Powered Energy Intelligence & Optimization Platform
Streamlit MVP Web Application
"""

from __future__ import annotations

import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import (
    init_db,
    insert_readings,
    fetch_readings,
    get_facilities,
    get_date_range,
    get_record_count,
    get_record_counts_by_source,
    clear_demonstration_readings,
    delete_records_by_facility,
    delete_records_by_source,
    clear_all_readings,
    get_all_existing_keys,
    check_existing_keys,
    get_default_db_path,
)
from data_processing import (
    load_file_to_df,
    validate_and_clean_data,
    DEFAULT_TARIFF_INR_PER_KWH,
)
from analytics import (
    run_anomaly_detection,
    get_anomaly_summary,
    generate_consumption_forecast,
)
from recommendations import generate_recommendations

# Set page configuration
st.set_page_config(
    page_title="FLUXORA | Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, high-polish aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
        border: 1px solid #f97316;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-low {
        background-color: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid #3b82f6;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .rec-card {
        background: #1e293b;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .rec-card-critical { border-left-color: #ef4444 !important; }
    .rec-card-high { border-left-color: #f97316 !important; }
    .rec-card-medium { border-left-color: #eab308 !important; }
    .rec-card-low { border-left-color: #3b82f6 !important; }
    
    .status-pill-normal {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    
    .status-pill-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    
    .status-pill-low {
        background: rgba(6, 182, 212, 0.15);
        color: #06b6d4;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize SQLite Database on startup
init_db()


def render_header():
    """Renders the top branding hero banner."""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="brand-title">⚡ FLUXORA</h1>
                <div class="brand-subtitle">AI-Powered Energy Intelligence & Optimization Platform</div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    ● Local SQLite Engine Active
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render any flash messages surviving st.rerun()
    if "flash_message" in st.session_state:
        st.success(st.session_state.pop("flash_message"))
    if "flash_error" in st.session_state:
        st.error(st.session_state.pop("flash_error"))


def sidebar_controls():
    """Renders the global sidebar configuration and navigation."""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Module",
            [
                "⚡ Energy Dashboard",
                "📥 Upload Data",
                "🔍 Anomalies & Insights",
                "🔮 7-Day Forecast",
                "💡 Recommendations",
                "🗄️ Data Explorer",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Analytical Settings")

        tariff = st.number_input(
            "Effective Tariff (₹ / kWh)",
            min_value=1.0,
            max_value=50.0,
            value=float(DEFAULT_TARIFF_INR_PER_KWH),
            step=0.1,
            help="Default electricity rate in INR used to calculate missing costs and financial impact.",
        )

        anomaly_threshold = st.slider(
            "Anomaly Deviation Threshold (%)",
            min_value=5.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Percentage deviation from historical baseline above or below which a reading is flagged.",
        )

        baseline_window = st.slider(
            "Baseline Rolling Window (Days)",
            min_value=7,
            max_value=30,
            value=14,
            step=1,
            help="Number of preceding days used to compute historical median baseline.",
        )

        st.markdown("---")
        counts = get_record_counts_by_source()
        st.markdown(f"**Database Status:** `{counts['total']}` records stored")
        if counts["total"] > 0:
            st.caption(f"• Demo: {counts['demo']} | Custom: {counts['user']}")
        st.caption(f"📁 SQLite DB: `{get_default_db_path()}`")

        st.markdown("---")
        st.caption("FLUXORA v1.0.0 (MVP) • Pune Campus Prototype")
        st.caption("Synthetic demonstration data • No external API required")

    return page, tariff, anomaly_threshold, baseline_window


# -----------------------------------------------------------------------------
# PAGE A: Energy Dashboard
# -----------------------------------------------------------------------------
def render_dashboard(tariff: float, anomaly_threshold: float, baseline_window: int):
    st.subheader("⚡ Energy Performance Dashboard")

    all_data = fetch_readings()

    if all_data.empty:
        st.info("👋 Welcome to FLUXORA! No energy records are currently loaded in the database.")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Quick Start Options:")
            st.markdown("""
            1. Go to **📥 Upload Data** to upload your own CSV or Excel (`.xlsx`) file.
            2. Or click the button below to instantly load the **Pune Campus 1-Year Demo Dataset (1,095 records)**.
            """)
            if st.button("🚀 Load Pune Campus Demo Dataset Now", type="primary", use_container_width=True):
                demo_csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "demo_energy.csv")
                if os.path.exists(demo_csv_path):
                    demo_df = pd.read_csv(demo_csv_path)
                    valid_df, _, _ = validate_and_clean_data(demo_df, effective_tariff=tariff)
                    inserted, _ = insert_readings(valid_df, data_source="demo_seed")
                    st.success(f"Successfully loaded {inserted} demonstration records into SQLite!")
                    st.rerun()
                else:
                    st.error("Demo file not found in sample_data directory.")
        return

    # Filter Bar
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    facilities = ["All Facilities"] + get_facilities()
    with col_f1:
        selected_facility = st.selectbox("Select Facility", facilities, index=0)

    min_d, max_d = get_date_range(selected_facility)
    with col_f2:
        start_date = st.date_input("Start Date", value=pd.to_datetime(min_d) if min_d else None)
    with col_f3:
        end_date = st.date_input("End Date", value=pd.to_datetime(max_d) if max_d else None)

    # Fetch filtered data from SQLite
    df = fetch_readings(
        facility_name=selected_facility,
        start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
        end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
    )

    if df.empty:
        st.warning("No records match the selected date and facility filters.")
        return

    # Run analytical pipeline
    analyzed_df = run_anomaly_detection(
        df,
        threshold_pct=anomaly_threshold,
        window_days=baseline_window,
        min_periods=max(3, baseline_window // 2),
    )

    # KPI Calculations
    total_kwh = float(df["consumption_kwh"].sum())
    total_cost = float(df["cost_inr"].sum())
    avg_daily_kwh = float(df["consumption_kwh"].mean())
    days_count = df["date"].nunique()
    est_cost_count = int(df.get("is_cost_estimated", 0).sum())
    est_cost_pct = (est_cost_count / len(df) * 100.0) if len(df) > 0 else 0.0

    # Period-over-period comparison (split in half)
    if len(df) >= 4:
        mid_pt = len(df) // 2
        p1_avg = df.iloc[:mid_pt]["consumption_kwh"].mean()
        p2_avg = df.iloc[mid_pt:]["consumption_kwh"].mean()
        period_change_pct = ((p2_avg - p1_avg) / p1_avg * 100.0) if p1_avg > 0 else 0.0
    else:
        period_change_pct = 0.0

    anomaly_sum = get_anomaly_summary(analyzed_df)

    # Render Metric Tiles
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Energy</div>
            <div class="metric-val">{total_kwh:,.1f}</div>
            <div class="metric-sub">kWh ({days_count} days)</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        cost_tag = f"{est_cost_pct:.0f}% est." if est_cost_pct > 0 else "Recorded"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-val">₹{total_cost:,.2f}</div>
            <div class="metric-sub">{cost_tag} @ ₹{tariff}/kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Average</div>
            <div class="metric-val">{avg_daily_kwh:,.1f}</div>
            <div class="metric-sub">kWh / day</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        change_color = "#ef4444" if period_change_pct > 0 else "#10b981"
        arrow = "▲" if period_change_pct > 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Period Trend</div>
            <div class="metric-val" style="color: {change_color};">{arrow} {abs(period_change_pct):.1f}%</div>
            <div class="metric-sub">Latest vs Earlier Half</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        data_health = 100.0 - min(100.0, anomaly_sum["anomaly_rate_pct"])
        health_color = "#10b981" if data_health >= 90 else "#eab308"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quality Score</div>
            <div class="metric-val" style="color: {health_color};">{data_health:.1f}%</div>
            <div class="metric-sub">{anomaly_sum['total_anomalies']} anomalies flagged</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Aggregation & Trend Chart
    st.markdown("#### 📈 Consumption Trend & Energy Profiles")
    agg_col1, agg_col2 = st.columns([1, 3])
    with agg_col1:
        agg_level = st.radio("Aggregation Level", ["Daily", "Weekly", "Monthly"], horizontal=True)

    # Prepare time-series aggregation
    plot_df = df.copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date"])

    if agg_level == "Weekly":
        plot_df["period"] = plot_df["date_dt"].dt.to_period("W").apply(lambda r: r.start_time)
        grouped = plot_df.groupby(["period", "facility_name"], as_index=False)["consumption_kwh"].sum()
        x_col = "period"
    elif agg_level == "Monthly":
        plot_df["period"] = plot_df["date_dt"].dt.to_period("M").apply(lambda r: r.start_time)
        grouped = plot_df.groupby(["period", "facility_name"], as_index=False)["consumption_kwh"].sum()
        x_col = "period"
    else:
        grouped = plot_df.sort_values(by="date_dt")
        x_col = "date"

    fig = px.line(
        grouped,
        x=x_col,
        y="consumption_kwh",
        color="facility_name",
        markers=True if agg_level != "Daily" or len(grouped) < 60 else False,
        title=f"{agg_level} Electricity Consumption (kWh) — {selected_facility}",
        labels={"consumption_kwh": "Energy Consumption (kWh)", x_col: "Time Period", "facility_name": "Facility"},
        color_discrete_sequence=["#10b981", "#06b6d4", "#f59e0b", "#a855f7", "#ec4899"],
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lower Grid: Facility Comparison & Day-of-Week Pattern
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🏢 Facility Share Comparison")
        fac_totals = df.groupby("facility_name", as_index=False)["consumption_kwh"].sum()
        fig_pie = px.pie(
            fac_totals,
            names="facility_name",
            values="consumption_kwh",
            hole=0.45,
            color_discrete_sequence=["#10b981", "#06b6d4", "#f59e0b", "#a855f7"],
            template="plotly_dark",
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#1e293b",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### 📅 Day-of-Week Consumption Profile")
        dow_df = df.copy()
        dow_df["date_dt"] = pd.to_datetime(dow_df["date"])
        dow_df["day_name"] = dow_df["date_dt"].dt.day_name()
        dow_df["day_idx"] = dow_df["date_dt"].dt.dayofweek
        dow_avg = dow_df.groupby(["day_idx", "day_name"], as_index=False)["consumption_kwh"].mean()
        dow_avg = dow_avg.sort_values(by="day_idx")

        fig_dow = px.bar(
            dow_avg,
            x="day_name",
            y="consumption_kwh",
            color="consumption_kwh",
            color_continuous_scale="Blues",
            labels={"day_name": "Day of Week", "consumption_kwh": "Avg Daily (kWh)"},
            template="plotly_dark",
        )
        fig_dow.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dow, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE B: Upload Data
# -----------------------------------------------------------------------------
def render_upload(tariff: float):
    st.subheader("📥 Data Ingestion & Validation")
    st.markdown("""
    Upload CSV or Excel (`.xlsx` / `.xls`) electricity meter readings.
    FLUXORA enforces strict date normalization, composite key validation (`facility_name` + `date`), in-file duplicate separation, and tariff estimation.
    """)

    db_path = get_default_db_path()
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px 16px; margin-bottom: 16px;">
        <span style="color: #94a3b8; font-size: 0.85rem;">🗄️ <b>Active SQLite Database File:</b></span> 
        <code style="color: #38bdf8; font-size: 0.85rem;">{db_path}</code>
    </div>
    """, unsafe_allow_html=True)

    # Demo dataset generator button & sample CSV download
    demo_csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "demo_energy.csv")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if os.path.exists(demo_csv_path):
            with open(demo_csv_path, "rb") as f:
                st.download_button(
                    label="📄 Download Sample Demo CSV (Pune Campus)",
                    data=f.read(),
                    file_name="fluxora_demo_energy.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with col_btn2:
        if st.button("🚀 Load Pune Campus Demo Dataset (365 Days) to DB", use_container_width=True):
            if os.path.exists(demo_csv_path):
                demo_df = pd.read_csv(demo_csv_path)
                valid_df, _, _ = validate_and_clean_data(demo_df, effective_tariff=tariff)
                inserted, skipped = insert_readings(valid_df, data_source="demo_seed")
                st.session_state["flash_message"] = f"Successfully imported {inserted} demonstration records! ({skipped} duplicates skipped)."
                st.rerun()

    # Database Status & Maintenance controls on upload page
    db_counts = get_record_counts_by_source()
    if db_counts["total"] > 0:
        with st.expander(
            f"🗄️ Database Status: {db_counts['total']} Records Stored ({db_counts['demo']} Demo, {db_counts['user']} Custom) — Maintenance Options",
            expanded=False,
        ):
            st.markdown(
                f"**Current Records in SQLite:** `{db_counts['total']}` total readings "
                f"(`{db_counts['demo']}` demonstration seed, `{db_counts['user']}` custom uploaded)."
            )
            
            # Show facilities summary
            if db_counts.get("facilities"):
                fac_summary_str = " | ".join([f"**{fac}**: {cnt}" for fac, cnt in db_counts["facilities"].items()])
                st.caption(f"Stored facilities: {fac_summary_str}")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown("##### 🧹 Clear Demo Seed Data")
                st.caption(
                    "Deletes only positively identified demo records (`demo_seed`, demo campus facilities). "
                    "All custom/user records remain intact."
                )
                if db_counts["demo"] > 0:
                    if st.button("🧹 Clear Demo Seed Records", key="upload_clear_demo_btn"):
                        cleared = clear_demonstration_readings()
                        st.session_state["flash_message"] = f"Successfully cleared {cleared} demonstration records from SQLite."
                        st.rerun()
                else:
                    st.caption("No demonstration records currently present.")

            with col_m2:
                st.markdown("##### 🗑️ Clear by Facility")
                st.caption("Targeted deletion of stored records for a specific facility.")
                if db_counts.get("facilities"):
                    fac_options = list(db_counts["facilities"].keys())
                    selected_del_fac = st.selectbox("Select Facility to Clear", fac_options, key="del_fac_sel")
                    if st.button(f"Delete Records for '{selected_del_fac}'", key="upload_del_fac_btn"):
                        cleared = delete_records_by_facility(selected_del_fac)
                        st.session_state["flash_message"] = f"Successfully deleted {cleared} records for '{selected_del_fac}'."
                        st.rerun()

            with col_m3:
                st.markdown("##### 🚨 Clear All Stored Records")
                st.caption("Permanently deletes all energy readings from the database.")
                confirm_upload_clear = st.checkbox(
                    "I confirm permanent deletion of all stored records.",
                    key="upload_confirm_clear_all",
                )
                if st.button("🚨 Clear All Records", disabled=not confirm_upload_clear, key="upload_btn_clear_all"):
                    cleared = clear_all_readings()
                    st.session_state["flash_message"] = f"Permanently cleared all {cleared} records from SQLite."
                    st.rerun()

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose an Energy Readings File (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="Required columns: 'date', 'consumption_kwh'. Optional: 'facility_name', 'cost_inr', 'occupancy', 'area_m2'.",
    )

    if uploaded_file is not None:
        try:
            raw_df = load_file_to_df(uploaded_file, file_name=uploaded_file.name)
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            return

        st.markdown("#### 1. Uploaded File Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        # Check existing database keys to identify DB duplicates
        existing_keys = get_all_existing_keys()

        # Validate
        valid_df, invalid_df, summary = validate_and_clean_data(
            raw_df, effective_tariff=tariff, existing_db_keys=existing_keys
        )

        st.markdown("#### 2. Validation Summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Rows", summary["total_rows"])
        m2.metric("Valid Ready", summary["valid_rows"])
        m3.metric("DB Duplicates", summary["db_duplicate_rows"])
        m4.metric("In-File Dups", summary.get("in_file_duplicates", 0))
        m5.metric("In-File Conflicts", summary.get("conflicting_records", 0))
        
        malformed_count = summary["invalid_rows"] - summary.get("in_file_duplicates", 0) - summary.get("conflicting_records", 0) - summary["db_duplicate_rows"]
        m6.metric("Malformed / Errors", max(0, malformed_count))

        # Display Validation Messages
        for msg in summary["messages"]:
            st.caption(f"• {msg}")

        # Helpful notice & action buttons if records conflict with existing DB records
        if summary["db_duplicate_rows"] > 0:
            st.warning(
                f"⚠️ **{summary['db_duplicate_rows']} row(s)** in this file already exist in the SQLite database "
                f"(`{db_path}`) and cannot be re-imported to prevent double-counting."
            )
            conflicting_facs = summary.get("conflicting_facilities", [])
            if conflicting_facs:
                st.info(f"Conflicting facilities currently in database: **{', '.join(conflicting_facs)}**")
                st.markdown("##### 🎯 Targeted Re-Import Options:")
                col_cf1, col_cf2 = st.columns([1, 1])
                with col_cf1:
                    for c_fac in conflicting_facs:
                        if st.button(f"🗑️ Clear Existing Records for '{c_fac}' to Allow Re-Import", key=f"quick_clear_fac_{c_fac}"):
                            cleared = delete_records_by_facility(c_fac)
                            st.session_state["flash_message"] = f"Cleared {cleared} records for '{c_fac}'. You can now re-import the file."
                            st.rerun()
                with col_cf2:
                    if db_counts["demo"] > 0:
                        if st.button("🧹 Clear Demonstration Seed Records", key="quick_clear_demo_btn_sub"):
                            cleared = clear_demonstration_readings()
                            st.session_state["flash_message"] = f"Cleared {cleared} demonstration records."
                            st.rerun()

        tab_valid, tab_invalid = st.tabs([
            f"✅ Valid Records ({len(valid_df)})",
            f"⚠️ Rejected Records ({len(invalid_df)})",
        ])

        with tab_valid:
            if not valid_df.empty:
                # Check for notices
                has_notices = "notice" in valid_df.columns and valid_df["notice"].astype(str).str.len().gt(0).any()
                if has_notices:
                    st.info("ℹ️ Some valid records have review notices (e.g. 0.0 kWh consumption). They are preserved as valid.")
                st.dataframe(valid_df, use_container_width=True)

                # Import button
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"💾 Import Valid Records ({len(valid_df)}) to SQLite Database", type="primary", use_container_width=True):
                    inserted, skipped = insert_readings(valid_df, data_source=uploaded_file.name)
                    st.session_state["flash_message"] = f"Successfully imported {inserted} records into the SQLite database! ({skipped} duplicates skipped)"
                    st.rerun()
            else:
                st.warning("No valid records found in the uploaded file.")

        with tab_invalid:
            if not invalid_df.empty:
                st.error("The following rows failed validation and will not be imported into the database:")
                
                # Rejection category filter
                categories = ["All Rejections"]
                if "rejection_category" in invalid_df.columns:
                    cat_counts = invalid_df["rejection_category"].value_counts().to_dict()
                    categories += [f"{cat} ({count})" for cat, count in cat_counts.items()]

                selected_cat = st.selectbox("Filter Rejected Records by Category", categories, key="filter_rejected_cat")
                
                display_invalid_df = invalid_df.copy()
                if selected_cat != "All Rejections":
                    cat_name = selected_cat.split(" (")[0]
                    display_invalid_df = display_invalid_df[display_invalid_df["rejection_category"] == cat_name]

                # Format columns nicely
                cols_order = [c for c in ["row_number", "date", "facility_name", "consumption_kwh", "rejection_category", "error_reason"] if c in display_invalid_df.columns]
                other_cols = [c for c in display_invalid_df.columns if c not in cols_order and c != "validation_status"]
                final_cols = cols_order + other_cols

                st.dataframe(display_invalid_df[final_cols], use_container_width=True)
            else:
                st.success("Zero errors! All uploaded rows passed strict validation.")



# -----------------------------------------------------------------------------
# PAGE C: Anomalies & Insights
# -----------------------------------------------------------------------------
def render_anomalies(anomaly_threshold: float, baseline_window: int):
    st.subheader("🔍 Explainable Consumption Anomaly Detection")
    st.markdown("""
    FLUXORA detects consumption anomalies by calculating a causal rolling historical median baseline.
    No future information is used in detection.
    """)

    df = fetch_readings()
    if df.empty:
        st.info("No records in database. Please upload data or load the demo dataset.")
        return

    facilities = ["All Facilities"] + get_facilities()
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_facility = st.selectbox("Filter Facility", facilities, key="anom_fac")
    with col_sel2:
        thresh_local = st.slider("Sensitivity Threshold (%)", 5.0, 50.0, float(anomaly_threshold), step=1.0, key="local_thresh")

    filtered_df = df if selected_facility == "All Facilities" else df[df["facility_name"] == selected_facility]
    
    analyzed_df = run_anomaly_detection(
        filtered_df,
        threshold_pct=thresh_local,
        window_days=baseline_window,
        min_periods=max(3, baseline_window // 2),
    )

    summary = get_anomaly_summary(analyzed_df)

    # Summary KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Evaluated Days", summary["evaluated_records"])
    k2.metric("High Anomalies", summary["high_anomaly_count"])
    k3.metric("Low Drops", summary["low_anomaly_count"])
    k4.metric("Anomaly Rate", f"{summary['anomaly_rate_pct']:.1f}%")
    k5.metric("Peak Spike", f"+{summary['highest_spike_pct']:.1f}%")

    st.markdown("---")

    # Interactive Chart
    st.markdown("#### 📊 Anomaly Time-Series & Baseline Envelope")

    chart_facs = analyzed_df["facility_name"].unique()
    for fac in chart_facs:
        fac_data = analyzed_df[analyzed_df["facility_name"] == fac].sort_values(by="date")
        
        fig = go.Figure()

        # Shaded Upper/Lower Bound Envelope
        valid_bounds = fac_data.dropna(subset=["lower_bound_kwh", "upper_bound_kwh"])
        if not valid_bounds.empty:
            fig.add_trace(go.Scatter(
                x=valid_bounds["date"].tolist() + valid_bounds["date"].tolist()[::-1],
                y=valid_bounds["upper_bound_kwh"].tolist() + valid_bounds["lower_bound_kwh"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"Expected Variance (±{thresh_local:.0f}%)",
                hoverinfo="skip",
            ))

        # Expected Baseline Line
        fig.add_trace(go.Scatter(
            x=fac_data["date"],
            y=fac_data["expected_kwh"],
            mode="lines",
            name="Historical Baseline (Median)",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))

        # Actual Normal Consumption Line
        fig.add_trace(go.Scatter(
            x=fac_data["date"],
            y=fac_data["consumption_kwh"],
            mode="lines",
            name="Actual Consumption (kWh)",
            line=dict(color="#38bdf8", width=2),
        ))

        # High Anomalies Scatter
        high_anoms = fac_data[fac_data["anomaly_status"] == "HIGH_ANOMALY"]
        if not high_anoms.empty:
            fig.add_trace(go.Scatter(
                x=high_anoms["date"],
                y=high_anoms["consumption_kwh"],
                mode="markers",
                name="High Anomaly (Spike)",
                marker=dict(color="#ef4444", size=10, symbol="triangle-up", line=dict(color="#ffffff", width=1)),
                text=high_anoms["explanation"],
                hoverinfo="x+y+text",
            ))

        # Low Anomalies Scatter
        low_anoms = fac_data[fac_data["anomaly_status"] == "LOW_ANOMALY"]
        if not low_anoms.empty:
            fig.add_trace(go.Scatter(
                x=low_anoms["date"],
                y=low_anoms["consumption_kwh"],
                mode="markers",
                name="Low Anomaly (Drop)",
                marker=dict(color="#eab308", size=10, symbol="triangle-down", line=dict(color="#ffffff", width=1)),
                text=low_anoms["explanation"],
                hoverinfo="x+y+text",
            ))

        fig.update_layout(
            title=f"Baseline & Detected Anomalies — {fac}",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Anomaly Table
    st.markdown("#### 📋 Detected Anomaly Event Log")
    anomalies_only = analyzed_df[analyzed_df["anomaly_status"].isin(["HIGH_ANOMALY", "LOW_ANOMALY"])].copy()

    if not anomalies_only.empty:
        display_anoms = anomalies_only[[
            "date", "facility_name", "consumption_kwh", "expected_kwh", "deviation_pct", "anomaly_status", "explanation"
        ]].sort_values(by="date", ascending=False)
        
        st.dataframe(
            display_anoms.rename(columns={
                "date": "Date",
                "facility_name": "Facility",
                "consumption_kwh": "Actual (kWh)",
                "expected_kwh": "Baseline (kWh)",
                "deviation_pct": "Deviation (%)",
                "anomaly_status": "Status",
                "explanation": "Diagnostic Explanation",
            }),
            use_container_width=True,
        )

        csv_anom = display_anoms.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Anomaly Event Log (CSV)",
            data=csv_anom,
            file_name="fluxora_anomalies.csv",
            mime="text/csv",
        )
    else:
        st.success("No anomalies detected with current threshold settings.")


# -----------------------------------------------------------------------------
# PAGE D: 7-Day Forecast
# -----------------------------------------------------------------------------
def render_forecast(tariff: float):
    st.subheader("🔮 7-Day Consumption Forecasting")
    st.markdown("""
    FLUXORA computes an explainable 7-day numerical energy projection using moving-average baselines,
    day-of-week seasonality factors, and recent linear trend estimation.
    """)

    df = fetch_readings()
    if df.empty:
        st.info("No energy records stored. Please load data to generate forecasts.")
        return

    facilities = ["Campus Total"] + get_facilities()
    selected_facility = st.selectbox("Select Target Facility for Forecasting", facilities, index=0)

    # Generate forecast
    forecast_df, meta = generate_consumption_forecast(
        df,
        facility_name=selected_facility,
        forecast_days=7,
        effective_tariff=tariff,
    )

    if forecast_df.empty:
        st.error(meta.get("message", "Could not generate forecast."))
        return

    # Forecast Metrics
    f_total_kwh = float(forecast_df["predicted_kwh"].sum())
    f_total_cost = float(forecast_df["estimated_cost_inr"].sum())
    f_daily_avg = float(forecast_df["predicted_kwh"].mean())

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Projected 7-Day Energy", f"{f_total_kwh:,.1f} kWh")
    f2.metric("Projected 7-Day Cost", f"₹{f_total_cost:,.2f}")
    f3.metric("Projected Daily Avg", f"{f_daily_avg:,.1f} kWh/day")
    f4.metric("Algorithm Method", "Moving-Avg + Seasonality")

    st.markdown("---")

    # Plot Forecast
    st.markdown("#### 📈 Historical Context & 7-Day Forecast")

    if selected_facility == "Campus Total":
        hist_series = df.groupby("date", as_index=False)["consumption_kwh"].sum()
    else:
        hist_series = df[df["facility_name"] == selected_facility].sort_values(by="date")

    # Take last 30 historical days for clarity
    recent_hist = hist_series.tail(30).copy()

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=recent_hist["date"],
        y=recent_hist["consumption_kwh"],
        mode="lines+markers",
        name="Historical Actual (Last 30 Days)",
        line=dict(color="#38bdf8", width=2),
    ))

    # Uncertainty Band
    fig.add_trace(go.Scatter(
        x=forecast_df["date"].tolist() + forecast_df["date"].tolist()[::-1],
        y=forecast_df["upper_kwh"].tolist() + forecast_df["lower_kwh"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Estimated Uncertainty Range",
        hoverinfo="skip",
    ))

    # Connect historical to forecast visually
    last_hist_row = recent_hist.iloc[-1]
    bridge_x = [last_hist_row["date"], forecast_df.iloc[0]["date"]]
    bridge_y = [last_hist_row["consumption_kwh"], forecast_df.iloc[0]["predicted_kwh"]]
    fig.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines",
        line=dict(color="#10b981", width=2, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_kwh"],
        mode="lines+markers",
        name="7-Day Predicted (kWh)",
        line=dict(color="#10b981", width=3, dash="dash"),
        marker=dict(size=8, color="#10b981"),
    ))

    fig.update_layout(
        title=f"7-Day Energy Forecast with Uncertainty Envelope — {selected_facility}",
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table
    st.markdown("#### 📋 7-Day Day-by-Day Forecast Schedule")
    table_df = forecast_df[["date", "day_name", "predicted_kwh", "lower_kwh", "upper_kwh", "estimated_cost_inr"]].copy()
    st.dataframe(
        table_df.rename(columns={
            "date": "Date",
            "day_name": "Day of Week",
            "predicted_kwh": "Predicted (kWh)",
            "lower_kwh": "Estimated Min (kWh)",
            "upper_kwh": "Estimated Max (kWh)",
            "estimated_cost_inr": f"Est. Cost (₹ @ {tariff}/kWh)",
        }),
        use_container_width=True,
    )

    st.info(f"**Methodology Note:** {meta.get('limitation_note', '')}")


# -----------------------------------------------------------------------------
# PAGE E: Recommendations
# -----------------------------------------------------------------------------
def render_recommendations(tariff: float, anomaly_threshold: float, baseline_window: int):
    st.subheader("💡 Actionable Energy Optimization Recommendations")
    st.markdown("""
    FLUXORA's deterministic expert rules evaluate detected anomalies, baseload profiles, and seasonal trends
    to generate prioritized engineering actions with estimated savings.
    """)

    df = fetch_readings()
    if df.empty:
        st.info("No energy records stored. Load data to generate recommendations.")
        return

    # Analyze data
    analyzed_df = run_anomaly_detection(
        df,
        threshold_pct=anomaly_threshold,
        window_days=baseline_window,
        min_periods=max(3, baseline_window // 2),
    )

    recs = generate_recommendations(analyzed_df, effective_tariff=tariff)

    if not recs:
        st.success("🎉 No operational anomalies or efficiency issues detected! Campus energy consumption is within nominal baselines.")
        return

    # Cumulative potential savings
    tot_saved_kwh = sum(r["estimated_savings_kwh"] for r in recs)
    tot_saved_inr = sum(r["estimated_savings_inr"] for r in recs)

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Recommendations", len(recs))
    c2.metric("Potential Energy Savings", f"{tot_saved_kwh:,.1f} kWh")
    c3.metric("Potential Financial Savings", f"₹{tot_saved_inr:,.2f}")

    st.markdown("---")

    # Priority Filter
    priority_filter = st.radio(
        "Filter by Priority",
        ["All Priorities", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        horizontal=True,
    )

    for rec in recs:
        if priority_filter != "All Priorities" and rec["priority"] != priority_filter:
            continue

        p_class = rec["priority"].lower()
        badge_html = f"<span class='badge-{p_class}'>{rec['priority']} PRIORITY</span>"

        card_html = f"""
        <div class="rec-card rec-card-{p_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; margin-right: 8px;">{rec['id']} • {rec['facility']}</span>
                    <span style="background: rgba(255,255,255,0.08); color: #cbd5e1; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem;">{rec['category']}</span>
                </div>
                <div>{badge_html}</div>
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                {rec['finding']}
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 10px;">
                <strong>Evidence:</strong> {rec['supporting_evidence']}
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;">
                <span style="color: #10b981; font-weight: 700;">Suggested Engineering Action:</span>
                <span style="color: #e2e8f0; font-size: 0.9rem; display: block; margin-top: 4px;">{rec['suggested_action']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #64748b; margin-top: 6px;">
                <span><strong>Rationale:</strong> {rec['reason']}</span>
                <span style="color: #10b981; font-weight: 600;">Est. Impact: ~{rec['estimated_savings_kwh']:,.1f} kWh (₹{rec['estimated_savings_inr']:,.2f})</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE F: Data Explorer
# -----------------------------------------------------------------------------
def render_explorer():
    st.subheader("🗄️ Stored Energy Records Explorer")
    st.markdown("Inspect, search, filter, export, or manage persistent SQLite database entries.")

    df = fetch_readings()

    if df.empty:
        st.info("No energy records are currently stored in the database. Upload a file or load demo records to populate.")
    else:
        facilities = ["All Facilities"] + get_facilities()
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            sel_fac = st.selectbox("Facility Filter", facilities, key="exp_fac")
        with col2:
            min_d, max_d = get_date_range(sel_fac)
            exp_start = st.date_input("From Date", value=pd.to_datetime(min_d) if min_d else None, key="exp_s")
        with col3:
            exp_end = st.date_input("To Date", value=pd.to_datetime(max_d) if max_d else None, key="exp_e")

        filtered = fetch_readings(
            facility_name=sel_fac,
            start_date=exp_start.strftime("%Y-%m-%d") if exp_start else None,
            end_date=exp_end.strftime("%Y-%m-%d") if exp_end else None,
        )

        st.markdown(f"**Showing {len(filtered)} matching records:**")
        st.dataframe(filtered, use_container_width=True)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="fluxora_energy_records.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### ⚠️ Database Maintenance & Reset")
    counts = get_record_counts_by_source()
    st.markdown(
        f"**Database Status:** `{counts['total']}` total records stored "
        f"(`{counts['demo']}` demonstration records, `{counts['user']}` custom uploaded records)."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 🧹 Clear Demonstration Records Only")
        st.caption(
            "Safely removes only demonstration records (`demo_seed`, `demo_energy.csv`, `fluxora_demo_energy.csv`). "
            "Any custom/non-demo user records remain completely intact."
        )
        if counts["demo"] > 0:
            if st.button("🧹 Clear Demonstration Records", key="exp_clear_demo_btn"):
                cleared = clear_demonstration_readings()
                st.session_state["flash_message"] = f"Successfully cleared {cleared} demonstration records from SQLite."
                st.rerun()
        else:
            st.info("No demonstration records currently present in database.")

    with col_m2:
        st.markdown("##### 🚨 Clear All Stored Records")
        st.caption("Permanently deletes ALL energy readings from the SQLite database and resets sequence counters.")
        if counts["total"] > 0:
            confirm_check = st.checkbox(
                "I understand and confirm that I want to permanently delete all stored records.",
                key="exp_confirm_clear_all",
            )
            if st.button("🚨 Clear All Records", type="secondary", disabled=not confirm_check, key="exp_btn_clear_all"):
                cleared = clear_all_readings()
                st.session_state["flash_message"] = f"Permanently cleared all {cleared} records from SQLite."
                st.rerun()
        else:
            st.info("Database is already empty.")



# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main():
    render_header()
    page, tariff, anomaly_threshold, baseline_window = sidebar_controls()

    if page == "⚡ Energy Dashboard":
        render_dashboard(tariff, anomaly_threshold, baseline_window)
    elif page == "📥 Upload Data":
        render_upload(tariff)
    elif page == "🔍 Anomalies & Insights":
        render_anomalies(anomaly_threshold, baseline_window)
    elif page == "🔮 7-Day Forecast":
        render_forecast(tariff)
    elif page == "💡 Recommendations":
        render_recommendations(tariff, anomaly_threshold, baseline_window)
    elif page == "🗄️ Data Explorer":
        render_explorer()


if __name__ == "__main__":
    main()
