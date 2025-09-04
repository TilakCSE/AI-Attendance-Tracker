"""
insight.py

Analytics helpers for Project Insight:
- compute_today_summary: present today, trend vs last 4 same weekdays
- compute_risk_list: rank students by low attendance rate and recent absences
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Any, List, Tuple

import pandas as pd

from database import DatabaseManager


def compute_today_summary(db: DatabaseManager) -> Dict[str, Any]:
    """
    Returns:
        {
            'present_today': int,
            'unique_students_today': int,
            'avg_same_weekday_past4': float,
            'delta_vs_avg': float
        }
    """
    df = db.to_dataframe()
    if df.empty:
        return {
            "present_today": 0,
            "unique_students_today": 0,
            "avg_same_weekday_past4": 0.0,
            "delta_vs_avg": 0.0,
        }

    today = date.today().isoformat()
    weekday = date.today().weekday()

    # Today metrics
    df_today = df[(df["date"] == today) & (df["status"] == "Present")]
    present_today = int(len(df_today))
    uniq_today = int(df_today["student_id"].nunique())

    # Past 4 same-weekday averages
    same_weekdays: List[str] = []
    d = date.today()
    seen = 0
    while seen < 4:
        d = d - timedelta(days=7)
        same_weekdays.append(d.isoformat())
        seen += 1
    df_same = df[(df["date"].isin(same_weekdays)) & (df["status"] == "Present")]
    avg_same = float(df_same.groupby("date")["student_id"].nunique().mean() or 0.0)

    delta = float(uniq_today - avg_same)

    return {
        "present_today": present_today,
        "unique_students_today": uniq_today,
        "avg_same_weekday_past4": round(avg_same, 2),
        "delta_vs_avg": round(delta, 2),
    }


def compute_risk_list(db: DatabaseManager, top_n: int = 10) -> pd.DataFrame:
    """
    Return a small DataFrame with columns:
    student_id, student_name, attendance_rate, consecutive_absences, last_seen
    Ordered by highest risk (low rate, many consecutive absences, old last_seen)
    """
    roster = db.roster_dataframe(deidentify=False)
    if roster.empty:
        return pd.DataFrame(columns=["student_id", "student_name", "attendance_rate", "consecutive_absences", "last_seen", "risk_score"])

    # risk score: weighted sum
    # Higher consecutive absences and lower attendance rate => higher risk
    roster["rate_inv"] = 1.0 - roster["attendance_rate"]
    roster["abs_weight"] = roster["consecutive_absences"].fillna(0).astype(float) / 10.0
    roster["risk_score"] = roster["rate_inv"] * 0.6 + roster["abs_weight"] * 0.4

    roster = roster.sort_values(by=["risk_score"], ascending=False)
    cols = ["student_id", "student_name", "attendance_rate", "consecutive_absences", "last_seen", "risk_score"]
    return roster[cols].head(top_n).reset_index(drop=True)
