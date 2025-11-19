import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import pickle
import os

def fine_tuned_forecast(logs_path='logs/user_cycle_log.csv', base_path='dataset/personal_cycle_logs_filled.csv'):
    """
    Fine-tune forecast using personal logs combined with base dataset.
    Returns dict with personalized forecast details.
    """
    # Load base dataset
    base_df = pd.read_csv(base_path)
    base_df['Start_Date'] = pd.to_datetime(base_df['Start_Date'], errors='coerce')
    base_df['End_Date'] = pd.to_datetime(base_df['End_Date'], errors='coerce')
    base_df = base_df.dropna(subset=['Start_Date', 'End_Date'])
    base_df['cycle_length'] = (base_df['End_Date'] - base_df['Start_Date']).dt.days + 1  # Approximate

    # Load personal logs
    if os.path.exists(logs_path):
        logs_df = pd.read_csv(logs_path)
        logs_df['date_start'] = pd.to_datetime(logs_df['date_start'], errors='coerce')
        logs_df['date_end'] = pd.to_datetime(logs_df['date_end'], errors='coerce')
        logs_df = logs_df.dropna(subset=['date_start', 'date_end'])
        logs_df['cycle_length'] = (logs_df['date_end'] - logs_df['date_start']).dt.days + 1
        personal_cycles = logs_df['cycle_length'].dropna().tolist()
    else:
        personal_cycles = []

    # Combine data
    all_cycles = base_df['cycle_length'].tolist() + personal_cycles
    if not all_cycles:
        return {"error": "Insufficient data for forecast"}

    # Simple fine-tuned forecast: weighted average (more weight to personal)
    base_avg = np.mean(base_df['cycle_length'])
    if personal_cycles:
        personal_avg = np.mean(personal_cycles)
        fine_tuned_cycle_length = 0.3 * base_avg + 0.7 * personal_avg  # Weight personal more
    else:
        fine_tuned_cycle_length = base_avg

    # Forecast next period (assume last personal or base)
    if personal_cycles:
        last_date = logs_df['date_start'].max()
    else:
        last_date = base_df['Start_Date'].max()

    predicted_start = last_date + timedelta(days=int(fine_tuned_cycle_length))
    next_start = (predicted_start - timedelta(days=2)).strftime('%b %d')
    next_end = (predicted_start + timedelta(days=2)).strftime('%b %d')

    return {
        "fine_tuned_cycle_length": round(fine_tuned_cycle_length, 1),
        "next_start": next_start,
        "next_end": next_end
    }

def compare_forecast_accuracy(logs_path='logs/user_cycle_log.csv', base_path='dataset/personal_cycle_logs_filled.csv'):
    """
    Compare baseline (base dataset only) vs fine-tuned (personal + base) accuracy.
    Returns dict with accuracy improvement percentage.
    """
    # Load base
    base_df = pd.read_csv(base_path)
    base_df['Start_Date'] = pd.to_datetime(base_df['Start_Date'], errors='coerce')
    base_df['End_Date'] = pd.to_datetime(base_df['End_Date'], errors='coerce')
    base_df = base_df.dropna(subset=['Start_Date', 'End_Date'])
    base_df['cycle_length'] = (base_df['End_Date'] - base_df['Start_Date']).dt.days + 1

    # Load personal
    if os.path.exists(logs_path):
        logs_df = pd.read_csv(logs_path)
        logs_df['date_start'] = pd.to_datetime(logs_df['date_start'], errors='coerce')
        logs_df['date_end'] = pd.to_datetime(logs_df['date_end'], errors='coerce')
        logs_df = logs_df.dropna(subset=['date_start', 'date_end'])
        logs_df['cycle_length'] = (logs_df['date_end'] - logs_df['date_start']).dt.days + 1
        personal_cycles = logs_df['cycle_length'].dropna().tolist()
    else:
        personal_cycles = []

    if len(base_df) < 2 or len(personal_cycles) < 2:
        return {"accuracy_improvement_%": 0}

    # Baseline: predict using base avg
    base_avg = np.mean(base_df['cycle_length'])
    actual_personal = personal_cycles[1:]  # Next cycles as actual
    baseline_pred = [base_avg] * len(actual_personal)
    baseline_mae = mean_absolute_error(actual_personal, baseline_pred) if actual_personal else None

    # Fine-tuned: weighted avg
    personal_avg = np.mean(personal_cycles)
    fine_tuned_avg = 0.3 * base_avg + 0.7 * personal_avg
    fine_tuned_pred = [fine_tuned_avg] * len(actual_personal)
    fine_tuned_mae = mean_absolute_error(actual_personal, fine_tuned_pred) if actual_personal else None

    if baseline_mae and fine_tuned_mae and baseline_mae > 0:
        accuracy_improvement_pct = ((baseline_mae - fine_tuned_mae) / baseline_mae) * 100
    else:
        accuracy_improvement_pct = 0

    return {
        "accuracy_improvement_%": round(accuracy_improvement_pct, 1)
    }
