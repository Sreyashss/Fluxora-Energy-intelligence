"""
FLUXORA Database Layer
Handles persistent SQLite storage for energy consumption readings with safety,
indexing, parameterization, and transaction control.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Set, Tuple, List, Dict
import pandas as pd

DEFAULT_DB_NAME = "fluxora.db"


def get_default_db_path() -> str:
    """Returns the default absolute path for the SQLite database."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DEFAULT_DB_NAME)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns a SQLite connection with row factory enabled
    and foreign keys turned on.
    """
    path = db_path or get_default_db_path()
    conn = sqlite3.connect(path, timeout=20.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes the database schema if tables do not exist.
    Creates tables and performance indexes safely.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS energy_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facility_name TEXT NOT NULL,
                date TEXT NOT NULL,
                consumption_kwh REAL NOT NULL,
                cost_inr REAL,
                is_cost_estimated INTEGER DEFAULT 0,
                occupancy REAL,
                area_m2 REAL,
                units_produced REAL,
                data_source TEXT DEFAULT 'upload',
                import_timestamp TEXT NOT NULL,
                CONSTRAINT uq_facility_date UNIQUE (facility_name, date)
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_facility_date 
            ON energy_readings (facility_name, date);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON energy_readings (date);
        """)
        conn.commit()


def check_existing_keys(
    keys: List[Tuple[str, str]], db_path: Optional[str] = None
) -> Set[Tuple[str, str]]:
    """
    Given a list of (facility_name, date) tuples, checks which ones
    already exist in the SQLite database to prevent redundant inserts.
    """
    if not keys:
        return set()

    init_db(db_path)
    existing: Set[Tuple[str, str]] = set()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Query in chunks if keys list is large
        chunk_size = 400
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join(["(?, ?)"] * len(chunk))
            flattened = [item for pair in chunk for item in pair]
            sql = f"""
                SELECT facility_name, date 
                FROM energy_readings 
                WHERE (facility_name, date) IN (VALUES {placeholders})
            """
            cursor.execute(sql, flattened)
            for row in cursor.fetchall():
                existing.add((str(row[0]), str(row[1])))

    return existing


def insert_readings(
    df: pd.DataFrame,
    data_source: str = "upload",
    db_path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Inserts validated energy readings from a pandas DataFrame into SQLite.
    
    Returns:
        (inserted_count, skipped_duplicate_count)
    """
    if df.empty:
        return 0, 0

    init_db(db_path)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Ensure required columns are present or populated with defaults
    required_cols = ["facility_name", "date", "consumption_kwh"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for database insertion.")

    # Prepare data records
    records_to_insert = []
    keys_to_check = []

    for _, row in df.iterrows():
        facility = str(row.get("facility_name", "Main Facility")).strip()
        date_str = str(row["date"]).strip()
        kwh = float(row["consumption_kwh"])
        cost = float(row["cost_inr"]) if pd.notnull(row.get("cost_inr")) else None
        is_est = int(row.get("is_cost_estimated", 0))
        occupancy = float(row["occupancy"]) if pd.notnull(row.get("occupancy")) else None
        area = float(row["area_m2"]) if pd.notnull(row.get("area_m2")) else None
        units = float(row["units_produced"]) if pd.notnull(row.get("units_produced")) else None

        keys_to_check.append((facility, date_str))
        records_to_insert.append((
            facility,
            date_str,
            kwh,
            cost,
            is_est,
            occupancy,
            area,
            units,
            data_source,
            now_str,
        ))

    existing_keys = check_existing_keys(keys_to_check, db_path=db_path)

    to_insert = []
    skipped_count = 0
    for rec in records_to_insert:
        key = (rec[0], rec[1])
        if key in existing_keys:
            skipped_count += 1
        else:
            to_insert.append(rec)
            # Add to local set so we don't insert in-batch duplicates twice
            existing_keys.add(key)

    if not to_insert:
        return 0, skipped_count

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO energy_readings (
                facility_name, date, consumption_kwh, cost_inr,
                is_cost_estimated, occupancy, area_m2, units_produced,
                data_source, import_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, to_insert)
        conn.commit()

    return len(to_insert), skipped_count


def fetch_readings(
    facility_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches energy readings matching the optional filters and returns a clean DataFrame.
    """
    init_db(db_path)

    query = "SELECT * FROM energy_readings WHERE 1=1"
    params: List[str] = []

    if facility_name and facility_name != "All Facilities":
        query += " AND facility_name = ?"
        params.append(facility_name)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC, facility_name ASC"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def get_facilities(db_path: Optional[str] = None) -> List[str]:
    """Returns a list of all distinct facility names currently stored."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT facility_name FROM energy_readings ORDER BY facility_name ASC")
        rows = cursor.fetchall()
        return [r[0] for r in rows if r[0]]


def get_date_range(
    facility_name: Optional[str] = None, db_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (min_date, max_date) for all records or a specific facility."""
    init_db(db_path)
    query = "SELECT MIN(date), MAX(date) FROM energy_readings"
    params = []
    if facility_name and facility_name != "All Facilities":
        query += " WHERE facility_name = ?"
        params.append(facility_name)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        if row and row[0] and row[1]:
            return str(row[0]), str(row[1])
    return None, None


def get_record_count(db_path: Optional[str] = None) -> int:
    """Returns the total number of records in the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_readings")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


DEMO_CAMPUS_FACILITIES = {
    "academic block a",
    "central library & data center",
    "student hostels & mess",
}


def is_demo_record(data_source: Optional[str], facility_name: Optional[str] = None) -> bool:
    """
    Positively identifies demonstration seed data:
    - Explicit 'demo_seed' source tag, OR
    - Built-in campus demo facilities imported from demo CSV templates.
    Does NOT match user-uploaded files for other facilities even if their filename has 'demo'.
    """
    ds = str(data_source or "").strip().lower()
    fn = str(facility_name or "").strip().lower()

    if ds == "demo_seed":
        return True
    if fn in DEMO_CAMPUS_FACILITIES and ("demo_energy" in ds or ds == "demo_seed"):
        return True
    return False


def get_all_existing_keys(db_path: Optional[str] = None) -> Set[Tuple[str, str]]:
    """
    Returns all (normalized_facility_name_lower, date) pairs currently stored in SQLite.
    Trimmed and lowercased for reliable, consistent composite key matching.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT facility_name, date FROM energy_readings")
        rows = cursor.fetchall()
        return {(str(r[0]).strip().lower(), str(r[1]).strip()) for r in rows if r[0] and r[1]}


def get_record_counts_by_source(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns breakdown of record counts:
    {'total': int, 'demo': int, 'user': int, 'sources': dict, 'facilities': dict}
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_readings")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM energy_readings
            WHERE data_source = 'demo_seed'
               OR (
                   LOWER(facility_name) IN ('academic block a', 'central library & data center', 'student hostels & mess')
                   AND (data_source = 'demo_seed' OR LOWER(data_source) LIKE '%demo_energy%')
               )
        """)
        demo = cursor.fetchone()[0]
        user = total - demo

        cursor.execute("SELECT data_source, COUNT(*) FROM energy_readings GROUP BY data_source")
        sources = {str(r[0]): int(r[1]) for r in cursor.fetchall()}

        cursor.execute("SELECT facility_name, COUNT(*) FROM energy_readings GROUP BY facility_name")
        facilities = {str(r[0]): int(r[1]) for r in cursor.fetchall()}

        return {
            "total": total,
            "demo": demo,
            "user": user,
            "sources": sources,
            "facilities": facilities,
        }


def clear_demonstration_readings(db_path: Optional[str] = None) -> int:
    """
    Clears ONLY positively identified demonstration records:
    - data_source = 'demo_seed'
    - Campus demo facilities with demo source markers
    Preserves all user-uploaded datasets regardless of file naming.
    Returns the count of deleted rows.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM energy_readings
            WHERE data_source = 'demo_seed'
               OR (
                   LOWER(facility_name) IN ('academic block a', 'central library & data center', 'student hostels & mess')
                   AND (data_source = 'demo_seed' OR LOWER(data_source) LIKE '%demo_energy%')
               )
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute("""
                DELETE FROM energy_readings
                WHERE data_source = 'demo_seed'
                   OR (
                       LOWER(facility_name) IN ('academic block a', 'central library & data center', 'student hostels & mess')
                       AND (data_source = 'demo_seed' OR LOWER(data_source) LIKE '%demo_energy%')
                   )
            """)
            cursor.execute("SELECT COUNT(*) FROM energy_readings")
            remaining = cursor.fetchone()[0]
            if remaining == 0:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='energy_readings'")
            conn.commit()
        return count


def delete_records_by_facility(facility_name: str, db_path: Optional[str] = None) -> int:
    """
    Deletes records for a specific facility (case-insensitive match).
    Returns the count of deleted rows.
    """
    init_db(db_path)
    clean_fac = facility_name.strip()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_readings WHERE LOWER(facility_name) = LOWER(?)", [clean_fac])
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute("DELETE FROM energy_readings WHERE LOWER(facility_name) = LOWER(?)", [clean_fac])
            cursor.execute("SELECT COUNT(*) FROM energy_readings")
            remaining = cursor.fetchone()[0]
            if remaining == 0:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='energy_readings'")
            conn.commit()
        return count


def delete_records_by_source(data_source: str, db_path: Optional[str] = None) -> int:
    """
    Deletes records matching a specific data source identifier.
    Returns the count of deleted rows.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_readings WHERE data_source = ?", [data_source])
        count = cursor.fetchone()[0]
        if count > 0:
            cursor.execute("DELETE FROM energy_readings WHERE data_source = ?", [data_source])
            cursor.execute("SELECT COUNT(*) FROM energy_readings")
            remaining = cursor.fetchone()[0]
            if remaining == 0:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='energy_readings'")
            conn.commit()
        return count


def clear_all_readings(db_path: Optional[str] = None) -> int:
    """
    Safely clears all records from the energy_readings table.
    Resets the auto-increment sequence.
    Returns the count of deleted rows.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM energy_readings")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM energy_readings")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='energy_readings'")
        conn.commit()
        return count


