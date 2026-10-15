"""
FLUXORA Analytics Engine
Provides explainable statistical anomaly detection and transparent 7-day
numerical energy consumption forecasting without external ML dependencies.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def detect_anomalies_for_facility(
    df: pd.DataFrame,
    threshold_pct: float = 15.0,
    window_days: int = 14,
    min_periods: int = 7
) -> pd.DataFrame:
    """
    Detects consumption anomalies for a single facility's chronological time-series.
    
    Uses a strictly causal rolling historical median baseline (shifted by 1 day)
    so no future information ever leaks into the baseline calculation.
    
    Returns a copy of the dataframe with added columns:
    - expected_kwh
    - deviation_pct
    - anomaly_status ('NORMAL', 'HIGH_ANOMALY', 'LOW_ANOMALY', 'INSUFFICIENT_DATA')
    - explanation
    - lower_bound_kwh
    - upper_bound_kwh
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "facility_name", "consumption_kwh", "expected_kwh",
            "deviation_pct", "anomaly_status", "explanation"
        ])

    data = df.sort_values(by="date").copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])

    # Strictly causal historical baseline: shift by 1 to exclude current day's reading
    shifted_series = data["consumption_kwh"].shift(1)
    rolling_median = shifted_series.rolling(window=window_days, min_periods=min_periods).median()
    
    # Rolling standard deviation / MAD for bounding
    rolling_std = shifted_series.rolling(window=window_days, min_periods=min_periods).std()
    # Fallback to a percentage of median if std is zero/null
    rolling_std = rolling_std.fillna(rolling_median * 0.1)

    expected_kwh_list: List[Optional[float]] = []
    deviation_pct_list: List[Optional[float]] = []
    anomaly_status_list: List[str] = []
    explanation_list: List[str] = []
    lower_bound_list: List[Optional[float]] = []
    upper_bound_list: List[Optional[float]] = []

    for idx, row in data.iterrows():
        actual = float(row["consumption_kwh"])
        exp = rolling_median.iloc[idx]
        std_val = rolling_std.iloc[idx]

        if pd.isna(exp) or exp is None:
            # Insufficient historical observations
            expected_kwh_list.append(None)
            deviation_pct_list.append(None)
            anomaly_status_list.append("INSUFFICIENT_DATA")
            explanation_list.append(
                f"Establishing baseline (requires at least {min_periods} prior days; observation #{idx + 1})."
            )
            lower_bound_list.append(None)
            upper_bound_list.append(None)
            continue

        exp_val = float(exp)
        
        # Handle zero baseline edge case
        if exp_val <= 0:
            if actual > 0:
                dev_pct = 100.0
                status = "HIGH_ANOMALY"
                explanation = f"Actual consumption ({actual:.1f} kWh) occurred when historical baseline was 0 kWh."
            else:
                dev_pct = 0.0
                status = "NORMAL"
                explanation = "Normal zero consumption baseline."
        else:
            dev_pct = ((actual - exp_val) / exp_val) * 100.0
            
            if dev_pct > threshold_pct:
                status = "HIGH_ANOMALY"
                explanation = (
                    f"Consumption of {actual:.1f} kWh is +{dev_pct:.1f}% above historical baseline "
                    f"({exp_val:.1f} kWh). Unusually high load detected."
                )
            elif dev_pct < -threshold_pct:
                status = "LOW_ANOMALY"
                explanation = (
                    f"Consumption of {actual:.1f} kWh is {dev_pct:.1f}% below historical baseline "
                    f"({exp_val:.1f} kWh). Unusually low load or shutdown detected."
                )
            else:
                status = "NORMAL"
                explanation = f"Within expected operating variance (deviation: {dev_pct:+.1f}% vs {threshold_pct:.0f}% threshold)."

        # Bounds for visualization
        bound_margin = exp_val * (threshold_pct / 100.0)
        lower_b = max(0.0, exp_val - bound_margin)
        upper_b = exp_val + bound_margin

        expected_kwh_list.append(round(exp_val, 2))
        deviation_pct_list.append(round(dev_pct, 2))
        anomaly_status_list.append(status)
        explanation_list.append(explanation)
        lower_bound_list.append(round(lower_b, 2))
        upper_bound_list.append(round(upper_b, 2))

    data["expected_kwh"] = expected_kwh_list
    data["deviation_pct"] = deviation_pct_list
    data["anomaly_status"] = anomaly_status_list
    data["explanation"] = explanation_list
    data["lower_bound_kwh"] = lower_bound_list
    data["upper_bound_kwh"] = upper_bound_list
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    return data


def run_anomaly_detection(
    df: pd.DataFrame,
    threshold_pct: float = 15.0,
    window_days: int = 14,
    min_periods: int = 7
) -> pd.DataFrame:
    """
    Applies anomaly detection across all facilities in the dataset.
    """
    if df.empty:
        return pd.DataFrame()

    results = []
    facilities = df["facility_name"].unique()

    for fac in facilities:
        fac_df = df[df["facility_name"] == fac].copy()
        fac_analyzed = detect_anomalies_for_facility(
            fac_df,
            threshold_pct=threshold_pct,
            window_days=window_days,
            min_periods=min_periods
        )
        results.append(fac_analyzed)

    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values(by=["date", "facility_name"]).reset_index(drop=True)


def get_anomaly_summary(analyzed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics from anomaly detection results.
    """
    if analyzed_df.empty or "anomaly_status" not in analyzed_df.columns:
        return {
            "total_records": 0,
            "evaluated_records": 0,
            "normal_count": 0,
            "high_anomaly_count": 0,
            "low_anomaly_count": 0,
            "total_anomalies": 0,
            "anomaly_rate_pct": 0.0,
            "insufficient_data_count": 0,
            "highest_spike_pct": 0.0,
            "lowest_drop_pct": 0.0,
        }

    total = len(analyzed_df)
    high_count = int((analyzed_df["anomaly_status"] == "HIGH_ANOMALY").sum())
    low_count = int((analyzed_df["anomaly_status"] == "LOW_ANOMALY").sum())
    normal_count = int((analyzed_df["anomaly_status"] == "NORMAL").sum())
    insufficient_count = int((analyzed_df["anomaly_status"] == "INSUFFICIENT_DATA").sum())
    evaluated = total - insufficient_count
    total_anomalies = high_count + low_count

    anomaly_rate = (total_anomalies / evaluated * 100.0) if evaluated > 0 else 0.0

    valid_devs = analyzed_df["deviation_pct"].dropna()
    highest_spike = float(valid_devs.max()) if not valid_devs.empty else 0.0
    lowest_drop = float(valid_devs.min()) if not valid_devs.empty else 0.0

    return {
        "total_records": total,
        "evaluated_records": evaluated,
        "normal_count": normal_count,
        "high_anomaly_count": high_count,
        "low_anomaly_count": low_count,
        "total_anomalies": total_anomalies,
        "anomaly_rate_pct": round(anomaly_rate, 1),
        "insufficient_data_count": insufficient_count,
        "highest_spike_pct": round(highest_spike, 1),
        "lowest_drop_pct": round(lowest_drop, 1),
    }


def generate_consumption_forecast(
    df: pd.DataFrame,
    facility_name: Optional[str] = None,
    forecast_days: int = 7,
    effective_tariff: float = 9.20,
    recent_window: int = 14
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a 7-day daily energy consumption forecast for a given facility.
    
    Algorithm:
    - Computes a historical baseline using the recent moving average (14-day window).
    - Incorporates day-of-week seasonality (ratios of each weekday vs weekly average).
    - Computes a conservative linear trend slope (least squares over recent window).
    - Computes empirical uncertainty bounds (± 1.96 * residual standard error or MAD).
    - Binds all predicted and lower-bound values strictly to >= 0.
    
    Returns:
        (forecast_df, method_metadata)
    """
    if df.empty:
        meta = {
            "status": "ERROR",
            "message": "Dataset is empty. Cannot generate forecast.",
            "method": "None",
            "sample_size": 0
        }
        return pd.DataFrame(), meta

    if facility_name and facility_name != "All Facilities":
        fac_data = df[df["facility_name"] == facility_name].copy()
    else:
        # If 'All Facilities' or None, aggregate total daily campus consumption
        fac_data = df.groupby("date", as_index=False)["consumption_kwh"].sum()
        facility_name = "Campus Total"

    if fac_data.empty:
        meta = {
            "status": "ERROR",
            "message": f"No data found for facility: '{facility_name}'",
            "method": "None",
            "sample_size": 0
        }
        return pd.DataFrame(), meta

    fac_data = fac_data.sort_values(by="date").reset_index(drop=True)
    fac_data["date_dt"] = pd.to_datetime(fac_data["date"])
    fac_data["day_of_week"] = fac_data["date_dt"].dt.day_name()
    fac_data["day_idx"] = fac_data["date_dt"].dt.dayofweek

    n_records = len(fac_data)
    last_date = fac_data["date_dt"].max()

    # Small dataset fallbacks
    if n_records < 3:
        # Extreme small data: simple naive repeat with high uncertainty
        avg_val = float(fac_data["consumption_kwh"].mean())
        forecast_rows = []
        for i in range(1, forecast_days + 1):
            f_date = last_date + pd.Timedelta(days=i)
            pred = max(0.0, avg_val)
            forecast_rows.append({
                "date": f_date.strftime("%Y-%m-%d"),
                "day_name": f_date.day_name(),
                "predicted_kwh": round(pred, 2),
                "lower_kwh": round(max(0.0, pred * 0.5), 2),
                "upper_kwh": round(pred * 1.5, 2),
                "estimated_cost_inr": round(pred * effective_tariff, 2),
                "facility_name": facility_name
            })
        meta = {
            "status": "WARNING",
            "method": "Simple Mean (Small Data Fallback)",
            "message": f"Only {n_records} historical records available. Forecast has wide uncertainty margins.",
            "sample_size": n_records,
            "trend_slope_kwh_per_day": 0.0,
            "mean_baseline_kwh": round(avg_val, 2)
        }
        return pd.DataFrame(forecast_rows), meta

    # Use recent window (min 7, max recent_window)
    window_size = min(n_records, max(7, recent_window))
    recent_slice = fac_data.tail(window_size).copy()

    # Baseline: recent mean
    baseline_mean = float(recent_slice["consumption_kwh"].mean())

    # Day of week seasonal index (ratio of each weekday's avg to total avg)
    dow_means = fac_data.groupby("day_idx")["consumption_kwh"].mean()
    overall_mean = float(fac_data["consumption_kwh"].mean())
    dow_ratios = {}
    for dow in range(7):
        if dow in dow_means and overall_mean > 0:
            ratio = dow_means[dow] / overall_mean
            # Dampen extreme ratios to stay stable
            dow_ratios[dow] = np.clip(ratio, 0.4, 2.0)
        else:
            dow_ratios[dow] = 1.0

    # Trend calculation using linear regression on recent slice
    x = np.arange(len(recent_slice))
    y = recent_slice["consumption_kwh"].values
    if len(x) >= 5 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        # Cap slope to ±5% of baseline per day to avoid divergence
        max_allowed_slope = baseline_mean * 0.05
        slope = float(np.clip(slope, -max_allowed_slope, max_allowed_slope))
    else:
        slope = 0.0

    # Uncertainty estimation: residual standard error or MAD
    if len(recent_slice) >= 3:
        fitted = intercept + slope * x if slope != 0 else np.full_like(y, baseline_mean)
        residuals = y - fitted
        residual_std = float(np.std(residuals))
        # Minimum margin of 5% of baseline
        margin = max(residual_std * 1.5, baseline_mean * 0.05)
    else:
        margin = baseline_mean * 0.15

    forecast_rows = []
    for step in range(1, forecast_days + 1):
        future_date = last_date + pd.Timedelta(days=step)
        future_dow_idx = future_date.dayofweek
        dow_factor = dow_ratios.get(future_dow_idx, 1.0)

        # Base value with trend step
        trended_base = baseline_mean + (slope * step)
        predicted = max(0.0, trended_base * dow_factor)
        
        # Uncertainty band grows slightly with forecast horizon
        horizon_factor = 1.0 + (step * 0.04)
        step_margin = margin * horizon_factor
        
        lower = max(0.0, predicted - step_margin)
        upper = predicted + step_margin
        cost_est = predicted * effective_tariff

        forecast_rows.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "day_name": future_date.day_name(),
            "predicted_kwh": round(predicted, 2),
            "lower_kwh": round(lower, 2),
            "upper_kwh": round(upper, 2),
            "estimated_cost_inr": round(cost_est, 2),
            "facility_name": facility_name
        })

    forecast_df = pd.DataFrame(forecast_rows)
    
    meta = {
        "status": "SUCCESS",
        "method": "Moving-Average with Day-of-Week Seasonality & Linear Trend",
        "sample_size": n_records,
        "window_days_used": window_size,
        "baseline_mean_kwh": round(baseline_mean, 2),
        "trend_slope_kwh_per_day": round(slope, 3),
        "estimated_total_forecast_kwh": round(float(forecast_df["predicted_kwh"].sum()), 2),
        "estimated_total_forecast_cost_inr": round(float(forecast_df["estimated_cost_inr"].sum()), 2),
        "limitation_note": (
            "Forecast represents statistical expected values based on recent historical patterns. "
            "Unscheduled events, weather extremes, equipment outages, or sudden campus occupancy changes "
            "are not reflected in numerical projections."
        )
    }

    return forecast_df, meta
