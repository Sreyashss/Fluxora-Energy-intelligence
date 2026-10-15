# FLUXORA — AI-Powered Energy Intelligence & Optimization Platform

FLUXORA is a lightweight, high-performance, and explainable **Energy Intelligence & Optimization MVP** designed for commercial, academic, and industrial campus facilities. It ingests electricity meter data, validates and stores readings in a local SQLite database, identifies consumption anomalies, generates 7-day numerical forecasts, and provides prioritized, actionable engineering recommendations.

---

## ⚡ Key Highlights & Core Capabilities

- **Zero External Dependencies / API-Free**: Entirely self-contained using Python, Streamlit, Pandas, NumPy, SQLite, and Plotly.
- **Robust Data Ingestion**: Supports `.csv`, `.xlsx`, and `.xls` files with flexible column alias matching, strict format validation, duplicate suppression, and automated tariff cost estimation.
- **Explainable Anomaly Detection**: Employs a strictly causal rolling historical median baseline (zero future leakage) with configurable deviation thresholds (default ±15%) to detect equipment overruns, scheduling errors, and data drops.
- **7-Day Numerical Forecasting**: Transparent, explainable projection combining rolling moving-average baselines, day-of-week seasonality ratios, and linear trend modeling with uncertainty envelopes.
- **Deterministic Recommendation Engine**: Rule-based expert system that translates empirical data spikes, weekend baseline leaks, and thermal drift into prioritized operational action plans with estimated savings.
- **Local Persistence**: Built on SQLite (`fluxora.db`) with unique `(facility_name, date)` constraints and indexed queries.

---

## 📁 Project Structure

```text
FLUXORA/
├── app.py                     # Streamlit web application & interactive UI
├── database.py                # SQLite persistence layer, schema & queries
├── data_processing.py         # File ingestion, data validation & cleaning
├── analytics.py               # Statistical anomaly detection & 7-day forecasting
├── recommendations.py         # Deterministic rule-based optimization engine
├── requirements.txt           # Core Python dependencies
├── README.md                  # System documentation & execution guide
├── sample_data/
│   └── demo_energy.csv        # 1-Year synthetic Pune Campus dataset (1,095 rows)
└── tests/
    └── test_analytics.py      # Comprehensive pytest verification suite (14 tests)
```

---

## 🚀 Quick Start & Installation (Windows PowerShell)

### 1. Prerequisites
- Python 3.10+ installed and added to `PATH`.

### 2. Navigate to Workspace
```powershell
cd D:\AI\FLUXORA\Fluxora
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Test Suite
Verify analytical correctness and edge-case resilience:
```powershell
pytest tests/test_analytics.py -v
```

### 5. Launch the Web Application
```powershell
streamlit run app.py
```
*The application will launch in your default web browser at `http://localhost:8501`.*

---

## 📊 Demonstration Walkthrough

1. **Energy Dashboard**:
   - On initial launch, click **"🚀 Load Pune Campus Demo Dataset Now"** to instantly populate the SQLite database with 1,095 realistic daily readings across 3 facilities (`Academic Block A`, `Central Library & Data Center`, `Student Hostels & Mess`).
   - Explore KPI tiles, period-over-period trend analysis, Daily/Weekly/Monthly aggregations, and Day-of-Week profiles.

2. **Upload & Validation**:
   - Download the demo CSV template using the **"📄 Download Sample Demo CSV"** button.
   - Upload any CSV or Excel file to observe real-time row-by-row validation, schema matching, error breakdown, and duplicate prevention.

3. **Anomalies & Insights**:
   - Adjust the **Sensitivity Threshold slider** (e.g. 15%) to see the baseline envelope adapt dynamically.
   - Inspect highlighted high spikes (e.g. Tech Fest overnight load on 2025-04-18, weekend chiller bypass on 2025-11-15) and drops (Diwali vacation on 2025-10-24).
   - Export the anomaly event log as CSV.

4. **7-Day Forecast**:
   - Select a facility or **Campus Total** to view the 7-day predicted energy consumption with upper and lower uncertainty bounds.
   - Review the daily schedule table with projected costs in INR (₹).

5. **Recommendations**:
   - Review prioritized recommendations (Critical, High, Medium, Low) generated from empirical data.
   - Inspect supporting evidence, suggested engineering actions, and potential kWh / INR financial ROI.

6. **Data Explorer & Database Maintenance**:
   - Filter, search, and export stored SQLite records.
   - Clear only demonstration records (preserving custom user uploads) or permanently clear all stored records with explicit confirmation.

---

## 🧪 Synthetic Dataset Disclaimer

The data provided in `sample_data/demo_energy.csv` represents reproducible, synthetic demonstration data modeled around typical Pune climate seasonality (summer peak cooling, mild monsoon, moderate winter) and academic facility operating profiles. It does not represent actual proprietary utility or university billing data.
