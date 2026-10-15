"""
FLUXORA Recommendation Engine
Deterministic, rule-based expert recommendation generator for campus and facility
energy optimization based on empirical analytical findings.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def generate_recommendations(
    analyzed_df: pd.DataFrame,
    effective_tariff: float = 9.20,
    facility_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates analyzed consumption data and anomaly flags to produce structured,
    prioritized, and actionable engineering recommendations.
    
    Each recommendation contains:
    - id
    - facility
    - priority ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    - category
    - finding
    - supporting_evidence
    - suggested_action
    - reason
    - estimated_savings_kwh
    - estimated_savings_inr
    """
    if analyzed_df.empty:
        return []

    data = analyzed_df.copy()
    if facility_filter and facility_filter != "All Facilities":
        data = data[data["facility_name"] == facility_filter]

    if data.empty:
        return []

    data["date_dt"] = pd.to_datetime(data["date"])
    facilities = data["facility_name"].unique()
    recommendations: List[Dict[str, Any]] = []
    rec_counter = 1

    for fac in facilities:
        fac_df = data[data["facility_name"] == fac].sort_values(by="date_dt").reset_index(drop=True)
        if len(fac_df) < 5:
            continue

        # Extract subsets
        high_anomalies = fac_df[fac_df["anomaly_status"] == "HIGH_ANOMALY"]
        low_anomalies = fac_df[fac_df["anomaly_status"] == "LOW_ANOMALY"]
        recent_30_days = fac_df.tail(30)
        recent_high_anomalies = recent_30_days[recent_30_days["anomaly_status"] == "HIGH_ANOMALY"]

        # -------------------------------------------------------------
        # Rule 1: Severe Single Spike (> 40% deviation)
        # -------------------------------------------------------------
        severe_spikes = fac_df[fac_df["deviation_pct"] >= 40.0]
        if not severe_spikes.empty:
            worst_spike = severe_spikes.loc[severe_spikes["deviation_pct"].idxmax()]
            worst_date = str(worst_spike["date"])
            worst_kwh = float(worst_spike["consumption_kwh"])
            worst_exp = float(worst_spike["expected_kwh"]) if pd.notnull(worst_spike["expected_kwh"]) else worst_kwh
            worst_dev = float(worst_spike["deviation_pct"])
            excess_kwh = max(0.0, worst_kwh - worst_exp)
            excess_cost = excess_kwh * effective_tariff

            recommendations.append({
                "id": f"REC-{rec_counter:03d}",
                "facility": fac,
                "priority": "CRITICAL" if worst_dev >= 75.0 else "HIGH",
                "category": "Operational Anomaly",
                "finding": f"Severe consumption spike detected at {fac} on {worst_date}.",
                "supporting_evidence": (
                    f"Measured {worst_kwh:.1f} kWh vs baseline expectation of {worst_exp:.1f} kWh "
                    f"(+{worst_dev:.1f}% deviation, +{excess_kwh:.1f} excess kWh on a single day)."
                ),
                "suggested_action": (
                    "Cross-reference facility logbook for scheduled events, heavy testing equipment, "
                    "or verify whether central cooling/heating plants were inadvertently kept running overnight."
                ),
                "reason": "Single large spikes usually indicate equipment left running after hours or operational schedule overrides.",
                "estimated_savings_kwh": round(excess_kwh, 1),
                "estimated_savings_inr": round(excess_cost, 2)
            })
            rec_counter += 1

        # -------------------------------------------------------------
        # Rule 2: Repeated High Anomalies (>= 3 in recent 30-day window)
        # -------------------------------------------------------------
        if len(recent_high_anomalies) >= 3:
            total_excess_kwh = float((recent_high_anomalies["consumption_kwh"] - recent_high_anomalies["expected_kwh"]).sum())
            monthly_est_waste = total_excess_kwh * effective_tariff

            recommendations.append({
                "id": f"REC-{rec_counter:03d}",
                "facility": fac,
                "priority": "HIGH",
                "category": "HVAC & Baseload",
                "finding": f"Chronic high-consumption anomalies ({len(recent_high_anomalies)} events in last 30 days).",
                "supporting_evidence": (
                    f"{len(recent_high_anomalies)} separate high-deviation days detected in the last 30 days, "
                    f"totaling ~{total_excess_kwh:.1f} kWh of cumulative excess consumption."
                ),
                "suggested_action": (
                    "Implement automated timer controls or smart building management schedules for major HVAC "
                    "units, pumps, and central lighting. Inspect chiller staging controls."
                ),
                "reason": "Recurrent high readings suggest systematic schedule drift or setpoint misalignment rather than isolated one-off events.",
                "estimated_savings_kwh": round(total_excess_kwh * 1.5, 1),
                "estimated_savings_inr": round(monthly_est_waste * 1.5, 2)
            })
            rec_counter += 1

        # -------------------------------------------------------------
        # Rule 3: Weekend Baseload Inefficiency
        # -------------------------------------------------------------
        fac_df["dow"] = fac_df["date_dt"].dt.dayofweek
        weekend_records = fac_df[fac_df["dow"].isin([5, 6])]  # Saturday, Sunday
        weekday_records = fac_df[fac_df["dow"].isin([0, 1, 2, 3, 4])]

        if not weekend_records.empty and not weekday_records.empty:
            avg_weekend = float(weekend_records["consumption_kwh"].mean())
            avg_weekday = float(weekday_records["consumption_kwh"].mean())

            # For Academic or Office blocks, weekend load should typically be < 45% of weekday
            if avg_weekday > 0:
                weekend_ratio = (avg_weekend / avg_weekday) * 100.0
                is_academic_or_office = any(term in fac.lower() for term in ["academic", "library", "admin", "office", "lab"])
                
                if is_academic_or_office and weekend_ratio > 55.0:
                    # High weekend baseload
                    est_weekend_reduction_kwh = (avg_weekend - (avg_weekday * 0.35)) * len(weekend_records)
                    est_weekend_reduction_kwh = max(0.0, est_weekend_reduction_kwh)
                    
                    recommendations.append({
                        "id": f"REC-{rec_counter:03d}",
                        "facility": fac,
                        "priority": "MEDIUM",
                        "category": "Baseload & Scheduling",
                        "finding": f"High weekend standby consumption at {fac} ({weekend_ratio:.1f}% of weekday level).",
                        "supporting_evidence": (
                            f"Weekend daily average is {avg_weekend:.1f} kWh vs weekday average of {avg_weekday:.1f} kWh "
                            f"(Ratio: {weekend_ratio:.1f}%, benchmark for academic buildings is < 40-45%)."
                        ),
                        "suggested_action": (
                            "Audit weekend baseline power consumption. Ensure non-critical laboratory chillers, "
                            "corridor lighting, and peripheral computer labs are powered down or put into standby mode."
                        ),
                        "reason": "Unoccupied academic facilities often waste substantial electricity maintaining active weekday equipment states.",
                        "estimated_savings_kwh": round(est_weekend_reduction_kwh / max(1, len(fac_df)/30), 1),
                        "estimated_savings_inr": round((est_weekend_reduction_kwh / max(1, len(fac_df)/30)) * effective_tariff, 2)
                    })
                    rec_counter += 1

        # -------------------------------------------------------------
        # Rule 4: Upward Drift / Trend Inefficiency
        # -------------------------------------------------------------
        if len(fac_df) >= 28:
            first_half_avg = float(fac_df.iloc[-28:-14]["consumption_kwh"].mean())
            second_half_avg = float(fac_df.iloc[-14:]["consumption_kwh"].mean())
            
            if first_half_avg > 0:
                drift_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100.0
                if drift_pct >= 15.0:
                    recommendations.append({
                        "id": f"REC-{rec_counter:03d}",
                        "facility": fac,
                        "priority": "MEDIUM",
                        "category": "Maintenance & Controls",
                        "finding": f"Sustained +{drift_pct:.1f}% consumption increase detected over the last 14 days.",
                        "supporting_evidence": (
                            f"14-day average increased from {first_half_avg:.1f} kWh to {second_half_avg:.1f} kWh "
                            f"without corresponding major changes in logged campus operations."
                        ),
                        "suggested_action": (
                            "Inspect air filters, condenser coils, and fan belts on primary AHUs and rooftop units. "
                            "Check thermostat calibrations for potential drift or sensor errors."
                        ),
                        "reason": "Gradual upward consumption drift frequently correlates with degraded heat transfer efficiency or dirty filters.",
                        "estimated_savings_kwh": round((second_half_avg - first_half_avg) * 14, 1),
                        "estimated_savings_inr": round((second_half_avg - first_half_avg) * 14 * effective_tariff, 2)
                    })
                    rec_counter += 1

        # -------------------------------------------------------------
        # Rule 5: Unusually Low Reading / Metering Integrity
        # -------------------------------------------------------------
        if len(low_anomalies) > 0:
            severe_drops = low_anomalies[low_anomalies["deviation_pct"] <= -50.0]
            if not severe_drops.empty:
                drop_record = severe_drops.iloc[-1]
                drop_date = str(drop_record["date"])
                drop_kwh = float(drop_record["consumption_kwh"])
                drop_exp = float(drop_record["expected_kwh"]) if pd.notnull(drop_record["expected_kwh"]) else 0.0

                recommendations.append({
                    "id": f"REC-{rec_counter:03d}",
                    "facility": fac,
                    "priority": "LOW",
                    "category": "Data Quality & Metering",
                    "finding": f"Significant negative consumption drop recorded at {fac} on {drop_date}.",
                    "supporting_evidence": (
                        f"Reading of {drop_kwh:.1f} kWh was {float(drop_record['deviation_pct']):.1f}% below expected {drop_exp:.1f} kWh."
                    ),
                    "suggested_action": (
                        "Verify whether a power outage, planned maintenance shutdown, or CT sensor communication failure "
                        "occurred on this date to ensure data integrity."
                    ),
                    "reason": "Extremely low readings often signal unmetered downtime or partial instrument disconnects.",
                    "estimated_savings_kwh": 0.0,
                    "estimated_savings_inr": 0.0
                })
                rec_counter += 1

    # Sort recommendations by priority order: CRITICAL -> HIGH -> MEDIUM -> LOW
    priority_weights = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_weights.get(x["priority"], 99))

    return recommendations
