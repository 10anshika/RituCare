import pandas as pd
from datetime import timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fine_tuning_model import fine_tuned_forecast

def get_next_period_prediction():
    """
    Generate the next period prediction for a single user (Anshika)
    by combining personal and recent log data for improved accuracy.
    """
    # Load datasets
    personal_df = pd.read_csv("dataset/personal_cycle_logs_filled.csv")
    logs_df = pd.read_csv("logs/user_cycle_log.csv")

    # Convert to datetime
    personal_df["Start_Date"] = pd.to_datetime(personal_df["Start_Date"], errors="coerce")
    personal_df["End_Date"] = pd.to_datetime(personal_df["End_Date"], errors="coerce")

    if "date_start" in logs_df.columns:
        logs_df["date_start"] = pd.to_datetime(logs_df["date_start"], errors="coerce")
    if "date_end" in logs_df.columns:
        logs_df["date_end"] = pd.to_datetime(logs_df["date_end"], errors="coerce")

    # Combine both datasets
    combined = pd.concat([
        personal_df.rename(columns={"Start_Date": "Start", "End_Date": "End"}),
        logs_df.rename(columns={"date_start": "Start", "date_end": "End"})
    ], ignore_index=True).dropna(subset=["Start", "End"])

    # Sort by Start date
    combined = combined.sort_values("Start").reset_index(drop=True)

    # Calculate cycle lengths (days between consecutive starts)
    cycle_lengths = []
    for i in range(len(combined) - 1):
        cycle_len = (combined.iloc[i+1]["Start"] - combined.iloc[i]["Start"]).days
        if cycle_len > 0:
            cycle_lengths.append(cycle_len)

    # Use hybrid fine-tuned forecast and personalize it
    base_forecast = fine_tuned_forecast()
    if cycle_lengths:
        user_avg = sum(cycle_lengths) / len(cycle_lengths)
        personalized_cycle = round((base_forecast["fine_tuned_cycle_length"] * 0.7 + user_avg * 0.3), 2)
    else:
        personalized_cycle = base_forecast["fine_tuned_cycle_length"]

    # Get last cycle end date
    last_end = combined["End"].max()

    # Predict next period
    next_start = last_end + timedelta(days=personalized_cycle)
    next_end = next_start + timedelta(days=5)

    return {
        "next_start": next_start.date(),
        "next_end": next_end.date(),
        "cycle_length": personalized_cycle
    }
