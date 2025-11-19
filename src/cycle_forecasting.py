import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def forecast_next_period(logs_df):
    """
    Forecast the next period start date based on historical cycle lengths.
    Returns a string representing a 5-day range around the predicted date.
    """
    if logs_df.empty or 'date_start' not in logs_df.columns:
        return "Log more periods to predict"

    # Convert date_start to datetime
    logs_df['date_start'] = pd.to_datetime(logs_df['date_start'], errors='coerce')

    # Drop rows with invalid dates
    logs_df = logs_df.dropna(subset=['date_start'])

    if len(logs_df) < 2:
        return "Log more periods to predict"

    # Sort by date_start
    logs_df = logs_df.sort_values('date_start')

    # Calculate cycle lengths
    cycle_lengths = []
    prev_date = None
    for date in logs_df['date_start']:
        if prev_date is not None:
            length = (date - prev_date).days
            if 20 <= length <= 45:  # Reasonable cycle length range
                cycle_lengths.append(length)
        prev_date = date

    if not cycle_lengths:
        return "Log more periods to predict"

    # Calculate average cycle length
    avg_cycle_length = np.mean(cycle_lengths)

    # Get the last period start date
    last_period_date = logs_df['date_start'].max()

    # Predict next period start
    predicted_start = last_period_date + timedelta(days=int(avg_cycle_length))

    # Create a 5-day range: 2 days before to 2 days after
    start_range = predicted_start - timedelta(days=2)
    end_range = predicted_start + timedelta(days=2)

    # Format as string
    range_str = f"{start_range.strftime('%b %d')} - {end_range.strftime('%b %d')}"

    return range_str

def predict_next_period(last_period_date, cycle_length):
    expected = last_period_date + timedelta(days=cycle_length)
    start_range = expected - timedelta(days=2)
    end_range = expected + timedelta(days=2)
    return expected, f"{start_range} — {end_range}"
