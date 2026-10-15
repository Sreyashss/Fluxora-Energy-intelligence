"""
FLUXORA Data Processing & Validation Layer
Handles file loading (CSV/Excel), schema normalization, strict data validation,
tariff estimation, duplicate detection, and audit logging for invalid rows.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple, Dict, Any, List, Union
import pandas as pd
import numpy as np

DEFAULT_TARIFF_INR_PER_KWH = 9.20

# Column mapping aliases to support various real-world headers
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "timestamp", "datetime", "reading_date", "day", "time"],
    "consumption_kwh": [
        "consumption_kwh", "kwh", "energy_kwh", "consumption", "energy_consumption",
        "electricity_kwh", "usage_kwh", "units_kwh", "daily_kwh"
    ],
    "facility_name": [
        "facility_name", "facility", "building", "site", "meter_id", "location", "block"
    ],
    "cost_inr": [
        "cost_inr", "cost", "cost_rs", "amount_inr", "bill_inr", "electricity_cost", "cost_rupees"
    ],
    "occupancy": ["occupancy", "occupants", "headcount", "people", "persons"],
    "area_m2": ["area_m2", "area", "area_sqm", "sq_m", "carpet_area_m2", "floor_area_m2"],
    "units_produced": ["units_produced", "production_units", "output_units", "production", "units"]
}


def normalize_column_name(col: str) -> str:
    """Normalizes column strings for fuzzy alias matching."""
    cleaned = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    for standard_name, aliases in COLUMN_ALIASES.items():
        if cleaned == standard_name or cleaned in aliases:
            return standard_name
    return cleaned


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames dataframe columns according to known alias mappings."""
    renamed_cols = {col: normalize_column_name(col) for col in df.columns}
    return df.rename(columns=renamed_cols)


def load_file_to_df(
    file_obj: Union[str, bytes, io.BytesIO, Any],
    file_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Reads a CSV or Excel (.xlsx / .xls) file into a pandas DataFrame.
    """
    name = file_name or getattr(file_obj, "name", "")
    name_lower = name.lower()

    if name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="openpyxl" if name_lower.endswith(".xlsx") else None)
    else:
        # Default to CSV parser with flexible delimiter detection
        try:
            df = pd.read_csv(file_obj)
        except Exception:
            # Retry with python engine for malformed delimiters
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, engine="python")

    return standardize_dataframe_columns(df)


def validate_and_clean_data(
    raw_df: pd.DataFrame,
    effective_tariff: float = DEFAULT_TARIFF_INR_PER_KWH,
    existing_db_keys: Optional[set] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive validation and cleaning on raw uploaded energy data.
    
    Returns:
        (valid_df, invalid_df, summary_dict)
    """
    if raw_df is None or raw_df.empty:
        summary = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "duplicate_rows": 0,
            "db_duplicate_rows": 0,
            "messages": ["The uploaded file contains no rows."]
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    df = standardize_dataframe_columns(raw_df.copy())
    total_rows = len(df)
    existing_db_keys = existing_db_keys or set()

    # Check for mandatory columns
    messages: List[str] = []
    if "date" not in df.columns:
        messages.append("Missing mandatory 'date' column.")
    if "consumption_kwh" not in df.columns:
        # Check if power was provided in W or kW without time conversion
        if "power_kw" in df.columns or "power_w" in df.columns:
            messages.append(
                "Found power columns (kW/W). Standard energy consumption must be specified in kWh. "
                "Please convert power multiplied by time (hours) to kWh before uploading."
            )
        else:
            messages.append("Missing mandatory 'consumption_kwh' column.")

    if messages:
        # Entire file failed schema check
        invalid_df = raw_df.copy()
        invalid_df["error_reason"] = "; ".join(messages)
        invalid_df["validation_status"] = "REJECTED"
        summary = {
            "total_rows": total_rows,
            "valid_rows": 0,
            "invalid_rows": total_rows,
            "duplicate_rows": 0,
            "db_duplicate_rows": 0,
            "messages": messages
        }
        return pd.DataFrame(), invalid_df, summary

    # Normalize existing database keys to (facility_lower, date) for reliable matching
    norm_db_keys = set()
    if existing_db_keys:
        norm_db_keys = {(str(f).strip().lower(), str(d).strip()) for f, d in existing_db_keys if f and d}

    # Row-by-row validation
    valid_rows = []
    invalid_rows = []
    seen_in_file: Dict[Tuple[str, str], Dict[str, Any]] = {}
    in_file_duplicate_count = 0
    in_file_conflict_count = 0
    db_duplicate_count = 0
    conflicting_db_facilities = set()

    for idx, row in df.iterrows():
        errors = []
        notices = []
        rejection_category = None
        raw_row_dict = raw_df.iloc[idx].to_dict()
        raw_row_dict["row_number"] = idx + 1

        # 1. Facility Name (Must NOT be missing or blank)
        raw_facility = row.get("facility_name")
        facility_name = None
        if pd.isna(raw_facility) or str(raw_facility).strip() == "":
            errors.append("Missing facility name: facility name cannot be blank or empty")
            rejection_category = "MALFORMED_DATA"
        else:
            facility_name = str(raw_facility).strip()

        # 2. Date Validation
        raw_date = row.get("date")
        parsed_date_str = None
        if pd.isna(raw_date) or str(raw_date).strip() == "":
            errors.append("Date is missing")
            rejection_category = "MALFORMED_DATA"
        else:
            try:
                # Support multiple date formats cleanly
                parsed_dt = pd.to_datetime(raw_date, dayfirst=False, errors="coerce")
                if pd.isna(parsed_dt):
                    # Try with dayfirst=True for Indian / UK standard dates DD-MM-YYYY
                    parsed_dt = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
                
                if pd.isna(parsed_dt):
                    errors.append(f"Invalid date format: '{raw_date}'")
                    rejection_category = "MALFORMED_DATA"
                else:
                    parsed_date_str = parsed_dt.strftime("%Y-%m-%d")
                    # Reasonable range check
                    if parsed_dt.year < 1990 or parsed_dt.year > 2050:
                        errors.append(f"Date year {parsed_dt.year} is out of realistic range (1990-2050)")
                        rejection_category = "MALFORMED_DATA"
            except Exception as e:
                errors.append(f"Date parsing error: {str(e)}")
                rejection_category = "MALFORMED_DATA"

        # 3. Consumption kWh Validation
        raw_kwh = row.get("consumption_kwh")
        clean_kwh = None
        if pd.isna(raw_kwh) or str(raw_kwh).strip() == "":
            errors.append("Energy consumption (kWh) is missing")
            rejection_category = "MALFORMED_DATA"
        else:
            raw_kwh_str = str(raw_kwh).strip().lower()
            # Explicit check for unit mismatch like MWh (do not silently convert)
            if "mwh" in raw_kwh_str:
                errors.append(f"Unit mismatch: found '{raw_kwh}' in kWh field. Conversion required before upload; not silently converted.")
                rejection_category = "MALFORMED_DATA"
            elif "mw" in raw_kwh_str or " kw" in raw_kwh_str or " w" in raw_kwh_str:
                errors.append(f"Unit mismatch: found power unit '{raw_kwh}'. Standard values must be in kWh.")
                rejection_category = "MALFORMED_DATA"
            else:
                try:
                    cleaned_str = raw_kwh_str.replace("kwh", "").replace(",", "").strip()
                    val_kwh = float(cleaned_str)
                    if np.isnan(val_kwh) or np.isinf(val_kwh):
                        errors.append("Energy consumption is NaN or Infinite")
                        rejection_category = "MALFORMED_DATA"
                    elif val_kwh < 0:
                        errors.append(f"Negative consumption rejected: {val_kwh} kWh")
                        rejection_category = "MALFORMED_DATA"
                    elif val_kwh == 0.0:
                        clean_kwh = 0.0
                        notices.append("Notice: Zero consumption reading (0.0 kWh) flagged for review.")
                    else:
                        clean_kwh = round(val_kwh, 3)
                except (ValueError, TypeError):
                    errors.append(f"Non-numeric energy consumption: '{raw_kwh}'")
                    rejection_category = "MALFORMED_DATA"

        # 4. Optional Fields Validation
        # Cost
        raw_cost = row.get("cost_inr")
        clean_cost = None
        is_cost_estimated = 0
        if pd.notnull(raw_cost) and str(raw_cost).strip() != "":
            try:
                cleaned_cost_str = str(raw_cost).lower().replace("rs", "").replace("inr", "").replace(",", "").replace("₹", "").strip()
                val_cost = float(cleaned_cost_str)
                if val_cost < 0:
                    errors.append(f"Cost cannot be negative: {val_cost}")
                    rejection_category = "MALFORMED_DATA"
                else:
                    clean_cost = round(val_cost, 2)
                    is_cost_estimated = 0
            except (ValueError, TypeError):
                errors.append(f"Non-numeric cost value: '{raw_cost}'")
                rejection_category = "MALFORMED_DATA"
        else:
            # Calculate estimated cost if consumption is valid
            if clean_kwh is not None:
                clean_cost = round(clean_kwh * effective_tariff, 2)
                is_cost_estimated = 1

        # Occupancy
        raw_occ = row.get("occupancy")
        clean_occ = None
        if pd.notnull(raw_occ) and str(raw_occ).strip() != "":
            try:
                val_occ = float(str(raw_occ).replace(",", "").strip())
                if val_occ < 0:
                    errors.append(f"Occupancy cannot be negative: {val_occ}")
                    rejection_category = "MALFORMED_DATA"
                else:
                    clean_occ = int(round(val_occ))
            except (ValueError, TypeError):
                errors.append(f"Non-numeric occupancy: '{raw_occ}'")
                rejection_category = "MALFORMED_DATA"

        # Area (m2)
        raw_area = row.get("area_m2")
        clean_area = None
        if pd.notnull(raw_area) and str(raw_area).strip() != "":
            try:
                val_area = float(str(raw_area).replace(",", "").strip())
                if val_area <= 0:
                    errors.append(f"Area must be greater than zero: {val_area}")
                    rejection_category = "MALFORMED_DATA"
                else:
                    clean_area = round(val_area, 2)
            except (ValueError, TypeError):
                errors.append(f"Non-numeric area: '{raw_area}'")
                rejection_category = "MALFORMED_DATA"

        # Units Produced
        raw_units = row.get("units_produced")
        clean_units = None
        if pd.notnull(raw_units) and str(raw_units).strip() != "":
            try:
                val_units = float(str(raw_units).replace(",", "").strip())
                if val_units < 0:
                    errors.append(f"Units produced cannot be negative: {val_units}")
                    rejection_category = "MALFORMED_DATA"
                else:
                    clean_units = round(val_units, 2)
            except (ValueError, TypeError):
                errors.append(f"Non-numeric units produced: '{raw_units}'")
                rejection_category = "MALFORMED_DATA"

        # 5. Composite Key Matching: In-File Duplicates, In-File Conflicts, and Database Duplicates
        if parsed_date_str and facility_name:
            composite_key = (facility_name.strip().lower(), parsed_date_str)
            # If the row already has fatal validation errors (e.g. negative consumption, non-numeric),
            # keep rejection_category as MALFORMED_DATA. Only evaluate duplicates for structurally valid readings.
            if not errors:
                if composite_key in seen_in_file:
                    prev_info = seen_in_file[composite_key]
                    prev_row_num = prev_info["row_index"] + 1
                    if clean_kwh is not None and prev_info.get("kwh") == clean_kwh:
                        in_file_duplicate_count += 1
                        errors.append(f"Exact in-file duplicate record of row {prev_row_num} (facility: '{facility_name}', date: '{parsed_date_str}')")
                        rejection_category = "IN_FILE_DUPLICATE"
                    else:
                        in_file_conflict_count += 1
                        errors.append(
                            f"In-file conflict: Duplicate record with conflicting data for facility '{facility_name}' on date '{parsed_date_str}' "
                            f"(row {prev_row_num} has {prev_info.get('kwh')} kWh vs this row's {clean_kwh} kWh)"
                        )
                        rejection_category = "IN_FILE_CONFLICT"
                elif composite_key in norm_db_keys:
                    db_duplicate_count += 1
                    errors.append(f"Record already exists in SQLite database for facility '{facility_name}' on date '{parsed_date_str}'")
                    rejection_category = "DB_DUPLICATE"
                    conflicting_db_facilities.add(facility_name)
                    seen_in_file[composite_key] = {
                        "row_index": idx,
                        "kwh": clean_kwh,
                        "facility": facility_name,
                        "date": parsed_date_str,
                        "in_db": True
                    }
                else:
                    seen_in_file[composite_key] = {
                        "row_index": idx,
                        "kwh": clean_kwh,
                        "facility": facility_name,
                        "date": parsed_date_str,
                        "in_db": False
                    }

        # Decide valid vs invalid
        if errors:
            raw_row_dict["error_reason"] = "; ".join(errors)
            raw_row_dict["rejection_category"] = rejection_category or "MALFORMED_DATA"
            raw_row_dict["validation_status"] = "REJECTED"
            invalid_rows.append(raw_row_dict)
        else:
            valid_row = {
                "row_number": idx + 1,
                "facility_name": facility_name,
                "date": parsed_date_str,
                "consumption_kwh": clean_kwh,
                "cost_inr": clean_cost,
                "is_cost_estimated": is_cost_estimated,
                "occupancy": clean_occ,
                "area_m2": clean_area,
                "units_produced": clean_units,
                "validation_status": "VALID",
                "notice": "; ".join(notices) if notices else ""
            }
            valid_rows.append(valid_row)

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    if not valid_df.empty:
        valid_df = valid_df.sort_values(by=["facility_name", "date"]).reset_index(drop=True)

    summary = {
        "total_rows": total_rows,
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "in_file_duplicates": in_file_duplicate_count,
        "conflicting_records": in_file_conflict_count,
        "db_duplicate_rows": db_duplicate_count,
        "conflicting_facilities": sorted(list(conflicting_db_facilities)),
        "messages": [
            f"Processed {total_rows} total rows.",
            f"Found {len(valid_rows)} valid records ready for import.",
            f"Found {len(invalid_rows)} rejected records: {in_file_duplicate_count} in-file duplicates, {in_file_conflict_count} in-file conflicts, {db_duplicate_count} existing DB duplicates, and {len(invalid_rows) - in_file_duplicate_count - in_file_conflict_count - db_duplicate_count} malformed rows."
        ]
    }

    return valid_df, invalid_df, summary

