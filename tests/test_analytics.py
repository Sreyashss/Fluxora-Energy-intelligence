"""
FLUXORA Test Suite
Verifies core analytical models, causal anomaly detection, 7-day numerical forecasting,
edge cases (zero baselines, small samples), and data validation rules.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics import (
    detect_anomalies_for_facility,
    run_anomaly_detection,
    get_anomaly_summary,
    generate_consumption_forecast,
)
from data_processing import validate_and_clean_data, DEFAULT_TARIFF_INR_PER_KWH


# -----------------------------------------------------------------------------
# 1. Anomaly Detection Tests
# -----------------------------------------------------------------------------

def test_normal_consumption_no_anomaly():
    """Consistent consumption within normal variance should remain NORMAL."""
    np.random.seed(101)
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    # Mean 500 kWh with tiny +/- 2% noise
    kwh = 500.0 + np.random.normal(0, 5.0, size=30)
    
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Test Facility",
        "consumption_kwh": kwh
    })

    result = detect_anomalies_for_facility(df, threshold_pct=15.0, window_days=14, min_periods=7)
    
    # After day 7 (index >= 7), all readings should be NORMAL
    evaluated_records = result.iloc[7:]
    for _, row in evaluated_records.iterrows():
        assert row["anomaly_status"] == "NORMAL", f"Row at {row['date']} misclassified as {row['anomaly_status']}"
        assert abs(row["deviation_pct"]) < 15.0


def test_spike_detection_with_history():
    """A deliberate +100% spike after baseline initialization must be flagged as HIGH_ANOMALY."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    kwh = [500.0] * 15 + [1000.0] + [500.0] * 4  # Day 15 is index 15 (16th day)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Test Facility",
        "consumption_kwh": kwh
    })

    result = detect_anomalies_for_facility(df, threshold_pct=15.0, window_days=14, min_periods=7)
    
    spike_row = result.iloc[15]
    assert spike_row["anomaly_status"] == "HIGH_ANOMALY"
    assert spike_row["expected_kwh"] == 500.0
    assert spike_row["deviation_pct"] == 100.0
    assert "Unusually high load detected" in spike_row["explanation"]


def test_low_consumption_drop_detection():
    """A deliberate -70% drop after baseline initialization must be flagged as LOW_ANOMALY."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    kwh = [600.0] * 15 + [150.0] + [600.0] * 4

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Test Facility",
        "consumption_kwh": kwh
    })

    result = detect_anomalies_for_facility(df, threshold_pct=15.0, window_days=14, min_periods=7)
    
    drop_row = result.iloc[15]
    assert drop_row["anomaly_status"] == "LOW_ANOMALY"
    assert drop_row["expected_kwh"] == 600.0
    assert drop_row["deviation_pct"] == -75.0
    assert "below historical baseline" in drop_row["explanation"]


def test_insufficient_history_handling():
    """Readings before min_periods threshold must be marked INSUFFICIENT_DATA without crashing."""
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    kwh = [500.0, 520.0, 510.0, 950.0, 500.0]

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Test Facility",
        "consumption_kwh": kwh
    })

    result = detect_anomalies_for_facility(df, threshold_pct=15.0, window_days=14, min_periods=7)
    
    assert len(result) == 5
    for status in result["anomaly_status"]:
        assert status == "INSUFFICIENT_DATA"


def test_zero_baseline_safe_division():
    """Zero baseline values must not cause ZeroDivisionError or crash."""
    dates = pd.date_range("2025-01-01", periods=15, freq="D")
    kwh = [0.0] * 10 + [50.0] * 5

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Test Facility",
        "consumption_kwh": kwh
    })

    # Should run smoothly without raising ZeroDivisionError
    result = detect_anomalies_for_facility(df, threshold_pct=15.0, window_days=14, min_periods=7)
    assert not result.empty
    assert len(result) == 15


# -----------------------------------------------------------------------------
# 2. Forecasting Tests
# -----------------------------------------------------------------------------

def test_forecast_seven_days():
    """Forecast must return exactly 7 consecutive future days with valid bounds."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    kwh = 600.0 + np.sin(np.linspace(0, 6, 30)) * 50.0

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Main Building",
        "consumption_kwh": kwh
    })

    forecast_df, meta = generate_consumption_forecast(df, facility_name="Main Building", forecast_days=7)
    
    assert len(forecast_df) == 7
    assert meta["status"] == "SUCCESS"
    assert "Moving-Average" in meta["method"]
    
    # Check consecutive future dates
    expected_start = pd.to_datetime("2025-01-31")
    for i, row in forecast_df.iterrows():
        expected_date = (expected_start + timedelta(days=i)).strftime("%Y-%m-%d")
        assert row["date"] == expected_date
        assert row["predicted_kwh"] >= 0.0
        assert row["lower_kwh"] <= row["predicted_kwh"]
        assert row["upper_kwh"] >= row["predicted_kwh"]
        assert row["lower_kwh"] >= 0.0


def test_forecast_empty_and_small_dataset():
    """Forecast must handle empty and minimal datasets gracefully."""
    # Empty
    empty_df = pd.DataFrame()
    f_empty, meta_empty = generate_consumption_forecast(empty_df)
    assert f_empty.empty
    assert meta_empty["status"] == "ERROR"

    # Minimal (2 records)
    small_df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "facility_name": ["Lab 1", "Lab 1"],
        "consumption_kwh": [300.0, 320.0]
    })
    f_small, meta_small = generate_consumption_forecast(small_df, facility_name="Lab 1", forecast_days=7)
    assert len(f_small) == 7
    assert meta_small["status"] == "WARNING"
    assert "Small Data Fallback" in meta_small["method"]


# -----------------------------------------------------------------------------
# 3. Data Ingestion & Validation Tests
# -----------------------------------------------------------------------------

def test_data_validation_clean():
    """Valid input rows must pass through cleanly."""
    raw = pd.DataFrame({
        "date": ["2025-01-01", "02/01/2025", "2025-01-03"],
        "facility_name": ["Admin", "Admin", "Admin"],
        "consumption_kwh": [500.0, 520.5, 490.2],
        "cost_inr": [4600.0, 4788.6, 4510.0],
        "occupancy": [150, 160, 140],
        "area_m2": [2500, 2500, 2500]
    })

    valid_df, invalid_df, summary = validate_and_clean_data(raw)
    assert len(valid_df) == 3
    assert len(invalid_df) == 0
    assert summary["valid_rows"] == 3
    assert summary["invalid_rows"] == 0


def test_data_validation_rejections():
    """Negative consumption and missing dates must be rejected with informative error reasons."""
    raw = pd.DataFrame({
        "date": ["2025-01-01", None, "2025-01-03", "2025-01-04"],
        "facility_name": ["Site A", "Site A", "Site A", "Site A"],
        "consumption_kwh": [500.0, 400.0, -150.0, "invalid_number"]
    })

    valid_df, invalid_df, summary = validate_and_clean_data(raw)
    assert len(valid_df) == 1  # Only row 0 is valid
    assert len(invalid_df) == 3
    assert "Date is missing" in invalid_df.iloc[0]["error_reason"]
    assert "Negative consumption" in invalid_df.iloc[1]["error_reason"]
    assert "Non-numeric" in invalid_df.iloc[2]["error_reason"]


def test_tariff_auto_estimation():
    """When cost is absent, calculate estimated cost with default tariff and flag is_cost_estimated=1."""
    raw = pd.DataFrame({
        "date": ["2025-01-01"],
        "facility_name": ["Site B"],
        "consumption_kwh": [1000.0]
        # cost_inr omitted
    })

    valid_df, _, _ = validate_and_clean_data(raw, effective_tariff=9.20)
    assert len(valid_df) == 1
    row = valid_df.iloc[0]
    assert row["cost_inr"] == 9200.0
    assert row["is_cost_estimated"] == 1


def test_duplicate_rejection():
    """In-file duplicate facility-date combinations must be flagged and rejected."""
    raw = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01"],
        "facility_name": ["Site C", "Site C"],
        "consumption_kwh": [500.0, 520.0]
    })

    valid_df, invalid_df, summary = validate_and_clean_data(raw)
    assert len(valid_df) == 1
    assert len(invalid_df) == 1
    assert "Duplicate record" in invalid_df.iloc[0]["error_reason"]


# -----------------------------------------------------------------------------
# 4. Database Persistence Tests
# -----------------------------------------------------------------------------

def test_database_lifecycle(tmp_path):
    """Database insertion, duplicate filtering, retrieval, and clear operations."""
    from database import init_db, insert_readings, fetch_readings, get_record_count, clear_all_readings, get_facilities

    test_db = str(tmp_path / "test_fluxora.db")
    init_db(test_db)
    assert get_record_count(test_db) == 0

    # Insert sample
    sample_df = pd.DataFrame({
        "facility_name": ["Block 1", "Block 1", "Block 2"],
        "date": ["2025-01-01", "2025-01-02", "2025-01-01"],
        "consumption_kwh": [500.0, 520.0, 800.0],
        "cost_inr": [4600.0, 4784.0, 7360.0],
        "is_cost_estimated": [0, 0, 0],
        "occupancy": [100, 110, 200],
        "area_m2": [1500.0, 1500.0, 2500.0],
        "units_produced": [None, None, None]
    })

    inserted, skipped = insert_readings(sample_df, db_path=test_db)
    assert inserted == 3
    assert skipped == 0
    assert get_record_count(test_db) == 3

    # Attempt re-inserting exact same data (duplicate prevention)
    inserted_2, skipped_2 = insert_readings(sample_df, db_path=test_db)
    assert inserted_2 == 0
    assert skipped_2 == 3
    assert get_record_count(test_db) == 3

    # Fetch with facility filter
    b1_df = fetch_readings(facility_name="Block 1", db_path=test_db)
    assert len(b1_df) == 2

    # Fetch facilities
    facs = get_facilities(test_db)
    assert "Block 1" in facs and "Block 2" in facs

    # Clear
    cleared = clear_all_readings(test_db)
    assert cleared == 3
    assert get_record_count(test_db) == 0


# -----------------------------------------------------------------------------
# 5. Recommendation Engine Tests
# -----------------------------------------------------------------------------

def test_recommendation_generation():
    """Rule-based recommendations should flag severe spikes and negative anomalies."""
    from recommendations import generate_recommendations

    dates = pd.date_range("2025-01-01", periods=25, freq="D")
    kwh = [500.0] * 20 + [1200.0] + [500.0] * 4  # Day 20 is a severe spike

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "facility_name": "Academic Block",
        "consumption_kwh": kwh
    })

    analyzed = detect_anomalies_for_facility(df, threshold_pct=15.0)
    recs = generate_recommendations(analyzed, effective_tariff=9.20)

    assert len(recs) > 0
    spike_rec = next((r for r in recs if "Severe consumption spike" in r["finding"]), None)
    assert spike_rec is not None
    assert spike_rec["priority"] in ["CRITICAL", "HIGH"]
    assert spike_rec["facility"] == "Academic Block"
    assert spike_rec["estimated_savings_kwh"] > 0
    assert "suggested_action" in spike_rec


# -----------------------------------------------------------------------------
# 6. Duplicate Import, Demo Clear, and Preservation Lifecycle Tests
# -----------------------------------------------------------------------------

def test_demo_dataset_import_duplicate_and_clear_lifecycle(tmp_path):
    """
    Verifies the full PBL dataset lifecycle:
    1. Importing 1,095 unique records from demo_energy.csv.
    2. Re-uploading the same dataset without creating duplicates.
    3. Preserving unrelated non-demo records when clearing demo records.
    4. Successfully clearing the intended demo records.
    5. Successfully re-importing the 1,095 records after clearing.
    6. Clearing all records with clear_all_readings.
    """
    from database import (
        init_db,
        insert_readings,
        fetch_readings,
        get_record_count,
        get_record_counts_by_source,
        clear_demonstration_readings,
        clear_all_readings,
        get_all_existing_keys,
    )
    from data_processing import load_file_to_df, validate_and_clean_data

    test_db = str(tmp_path / "lifecycle_fluxora.db")
    init_db(test_db)

    # Step 1: Load 1,095 records from demo_energy.csv
    demo_csv_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "demo_energy.csv")
    raw_df = load_file_to_df(demo_csv_path)
    assert len(raw_df) == 1095

    # First upload against empty DB
    existing_keys_1 = get_all_existing_keys(test_db)
    valid_df_1, invalid_df_1, summary_1 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_1)
    assert summary_1["valid_rows"] == 1095
    assert summary_1["db_duplicate_rows"] == 0

    inserted_1, skipped_1 = insert_readings(valid_df_1, data_source="fluxora_demo_energy.csv", db_path=test_db)
    assert inserted_1 == 1095
    assert skipped_1 == 0
    assert get_record_count(test_db) == 1095

    # Step 2: Re-uploading the same dataset without creating duplicates
    existing_keys_2 = get_all_existing_keys(test_db)
    valid_df_2, invalid_df_2, summary_2 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_2)
    assert summary_2["valid_rows"] == 0
    assert summary_2["db_duplicate_rows"] == 1095
    assert len(valid_df_2) == 0

    # Ensure insert_readings also rejects any duplicate attempt
    inserted_2, skipped_2 = insert_readings(valid_df_1, data_source="fluxora_demo_energy.csv", db_path=test_db)
    assert inserted_2 == 0
    assert skipped_2 == 1095
    assert get_record_count(test_db) == 1095  # No duplicates created

    # Step 3: Insert an unrelated custom/user record to test preservation
    user_record_df = pd.DataFrame([{
        "facility_name": "Campus Medical Center",
        "date": "2025-06-01",
        "consumption_kwh": 320.0,
        "cost_inr": 2944.0,
        "is_cost_estimated": 0,
        "occupancy": 45,
        "area_m2": 1200.0,
        "units_produced": None
    }])
    user_inserted, _ = insert_readings(user_record_df, data_source="hospital_meters.csv", db_path=test_db)
    assert user_inserted == 1
    assert get_record_count(test_db) == 1096

    counts = get_record_counts_by_source(test_db)
    assert counts["total"] == 1096
    assert counts["demo"] == 1095
    assert counts["user"] == 1

    # Step 4: Clear ONLY demonstration records
    cleared_demo = clear_demonstration_readings(test_db)
    assert cleared_demo == 1095

    # Step 5: Verify preservation of unrelated user records
    counts_after_demo_clear = get_record_counts_by_source(test_db)
    assert counts_after_demo_clear["total"] == 1
    assert counts_after_demo_clear["demo"] == 0
    assert counts_after_demo_clear["user"] == 1

    remaining_df = fetch_readings(db_path=test_db)
    assert len(remaining_df) == 1
    assert remaining_df.iloc[0]["facility_name"] == "Campus Medical Center"
    assert remaining_df.iloc[0]["data_source"] == "hospital_meters.csv"

    # Step 6: Successfully import the 1,095 records again after clearing demo records
    existing_keys_3 = get_all_existing_keys(test_db)
    valid_df_3, invalid_df_3, summary_3 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_3)
    assert summary_3["valid_rows"] == 1095
    assert summary_3["db_duplicate_rows"] == 0

    inserted_3, skipped_3 = insert_readings(valid_df_3, data_source="fluxora_demo_energy.csv", db_path=test_db)
    assert inserted_3 == 1095
    assert skipped_3 == 0
    # Total is now 1,095 demo + 1 preserved custom = 1,096
    assert get_record_count(test_db) == 1096

    # Step 7: Clear all records
    cleared_all = clear_all_readings(test_db)
    assert cleared_all == 1096
    assert get_record_count(test_db) == 0


def test_messy_manufacturing_dataset_validation():
    """
    Validates the messy manufacturing dataset (102 rows) ensuring:
    - 91 valid rows ready for import
    - 11 rejected rows: 1 in-file duplicate, 1 in-file conflict, 9 malformed rows
    - Missing facility is rejected (not defaulted to Main Campus)
    - 4.7 MWh unit mismatch is rejected (not silently converted)
    - 0.0 kWh consumption is accepted as valid with a review notice
    - Negative occupancy, negative area, negative consumption, non-numeric values rejected
    """
    from data_processing import load_file_to_df
    csv_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "fluxora_messy_manufacturing_test.csv")
    assert os.path.exists(csv_path), "Test file not found"

    raw_df = load_file_to_df(csv_path)
    assert len(raw_df) == 102

    valid_df, invalid_df, summary = validate_and_clean_data(raw_df)

    assert summary["total_rows"] == 102
    assert summary["valid_rows"] == 91
    assert summary["invalid_rows"] == 11
    assert summary["in_file_duplicates"] == 1
    assert summary["conflicting_records"] == 1
    assert summary["db_duplicate_rows"] == 0

    # Verify zero-consumption row (2026-04-02) is accepted as valid with notice
    zero_row = valid_df[valid_df["date"] == "2026-04-02"]
    assert len(zero_row) == 1
    assert zero_row.iloc[0]["consumption_kwh"] == 0.0
    assert "Notice: Zero consumption reading" in zero_row.iloc[0]["notice"]

    # Verify specific rejection reasons in invalid_df
    rejection_reasons = " ".join(invalid_df["error_reason"].tolist())

    # 1. Exact in-file duplicate
    dup_rows = invalid_df[invalid_df["rejection_category"] == "IN_FILE_DUPLICATE"]
    assert len(dup_rows) == 1
    assert "Exact in-file duplicate record" in dup_rows.iloc[0]["error_reason"]

    # 2. In-file conflict
    conflict_rows = invalid_df[invalid_df["rejection_category"] == "IN_FILE_CONFLICT"]
    assert len(conflict_rows) == 1
    assert "In-file conflict" in conflict_rows.iloc[0]["error_reason"]
    assert "99999" in conflict_rows.iloc[0]["error_reason"]

    # 3. Missing date
    assert "Date is missing" in rejection_reasons

    # 4. Invalid date
    assert "Invalid date format: '31/31/2026'" in rejection_reasons

    # 5. Negative consumption
    assert "Negative consumption rejected" in rejection_reasons

    # 6. Non-numeric consumption
    assert "Non-numeric energy consumption: 'four thousand'" in rejection_reasons

    # 7. Missing facility (must not be defaulted to Main Campus)
    assert "Missing facility name: facility name cannot be blank or empty" in rejection_reasons
    assert "Main Campus" not in valid_df["facility_name"].values

    # 8. Missing consumption
    assert "Energy consumption (kWh) is missing" in rejection_reasons

    # 9. Negative occupancy
    assert "Occupancy cannot be negative: -5" in rejection_reasons

    # 10. Unit mismatch (4.7 MWh in kWh field)
    assert "Unit mismatch: found '4.7 MWh'" in rejection_reasons

    # 11. Negative area
    assert "Area must be greater than zero: -2800" in rejection_reasons


def test_messy_manufacturing_sqlite_workflow(tmp_path):
    """
    End-to-end test verifying:
    1. A new dataset with no matching keys imports its valid rows.
    2. Re-uploading it detects existing duplicates separately from in-file issues.
    3. A different facility on the same date is NOT treated as a duplicate.
    4. Clearing demo records removes only demo records and preserves user records.
    5. Targeted facility deletion allows re-importing cleanly.
    6. Invalid rows do not prevent valid rows from being imported.
    """
    from database import (
        init_db,
        insert_readings,
        get_record_count,
        get_all_existing_keys,
        clear_demonstration_readings,
        delete_records_by_facility,
        get_record_counts_by_source,
    )
    from data_processing import load_file_to_df

    test_db = str(tmp_path / "manufacturing_test.db")
    init_db(test_db)
    assert get_record_count(test_db) == 0

    csv_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "fluxora_messy_manufacturing_test.csv")
    raw_df = load_file_to_df(csv_path)

    # Step 1: Initial upload - 91 valid rows, 11 rejected rows
    existing_keys_1 = get_all_existing_keys(test_db)
    assert len(existing_keys_1) == 0

    valid_df_1, invalid_df_1, summary_1 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_1)
    assert summary_1["valid_rows"] == 91
    assert summary_1["invalid_rows"] == 11
    assert summary_1["db_duplicate_rows"] == 0

    # Invalid rows do not prevent valid rows from being imported
    inserted_1, skipped_1 = insert_readings(valid_df_1, data_source="fluxora_messy_manufacturing_test.csv", db_path=test_db)
    assert inserted_1 == 91
    assert skipped_1 == 0
    assert get_record_count(test_db) == 91

    # Step 2: Re-uploading detects existing database duplicates
    existing_keys_2 = get_all_existing_keys(test_db)
    assert len(existing_keys_2) == 91

    valid_df_2, invalid_df_2, summary_2 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_2)
    assert summary_2["valid_rows"] == 0
    assert summary_2["invalid_rows"] == 102
    assert summary_2["db_duplicate_rows"] == 91
    assert summary_2["in_file_duplicates"] == 1
    assert summary_2["conflicting_records"] == 1

    # Step 3: A different facility on the same date is NOT treated as a duplicate
    diff_facility_df = pd.DataFrame([{
        "date": "2026-01-01",  # Same date as row 1 of Pune Precision Components
        "facility_name": "Nashik Foundry Plant",
        "consumption_kwh": 3500.0,
        "cost_inr": 32200.0,
    }])
    v_diff, inv_diff, s_diff = validate_and_clean_data(diff_facility_df, existing_db_keys=existing_keys_2)
    assert s_diff["valid_rows"] == 1
    assert s_diff["db_duplicate_rows"] == 0
    ins_diff, _ = insert_readings(v_diff, data_source="nashik_data.csv", db_path=test_db)
    assert ins_diff == 1
    assert get_record_count(test_db) == 92

    # Step 4: Add demo records and test clearing demo records preserves user records
    demo_df = pd.DataFrame([{
        "facility_name": "Academic Block A",
        "date": "2025-01-01",
        "consumption_kwh": 400.0,
    }])
    v_demo, _, _ = validate_and_clean_data(demo_df)
    insert_readings(v_demo, data_source="demo_seed", db_path=test_db)
    assert get_record_count(test_db) == 93

    counts_before = get_record_counts_by_source(test_db)
    assert counts_before["demo"] == 1
    assert counts_before["user"] == 92

    # Clearing demo records removes ONLY demo records and preserves user records
    cleared_demo = clear_demonstration_readings(test_db)
    assert cleared_demo == 1
    assert get_record_count(test_db) == 92

    counts_after = get_record_counts_by_source(test_db)
    assert counts_after["demo"] == 0
    assert counts_after["user"] == 92

    # Step 5: Delete Pune Precision Components records to allow re-import
    cleared_fac = delete_records_by_facility("Pune Precision Components", db_path=test_db)
    assert cleared_fac == 91
    assert get_record_count(test_db) == 1  # Only Nashik Foundry Plant remains

    # Step 6: Re-import works after clearing
    existing_keys_3 = get_all_existing_keys(test_db)
    valid_df_3, invalid_df_3, summary_3 = validate_and_clean_data(raw_df, existing_db_keys=existing_keys_3)
    assert summary_3["valid_rows"] == 91
    assert summary_3["db_duplicate_rows"] == 0

    inserted_3, skipped_3 = insert_readings(valid_df_3, data_source="fluxora_messy_manufacturing_test.csv", db_path=test_db)
    assert inserted_3 == 91
    assert skipped_3 == 0
    assert get_record_count(test_db) == 92



