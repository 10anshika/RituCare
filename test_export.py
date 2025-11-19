import pandas as pd
import numpy as np
import io
from datetime import datetime

# Load logs
logs = pd.read_csv('logs/user_cycle_log.csv')
logs['date_start'] = pd.to_datetime(logs['date_start'], errors='coerce')
logs['date_end'] = pd.to_datetime(logs['date_end'], errors='coerce')
logs['date'] = pd.to_datetime(logs['date'], errors='coerce')

# Clean the dataset
cleaned_df = logs.dropna(axis=1, how='all')
cleaned_df.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_df.columns]

# Separate period logs and water logs
period_columns = ['user_id', 'date_start', 'date_end', 'flow', 'mood', 'symptoms', 'notes']
available_period_columns = [col for col in period_columns if col in cleaned_df.columns]
period_logs = cleaned_df.dropna(subset=['date_start', 'date_end'])[available_period_columns]
water_logs_all = cleaned_df[cleaned_df['date_start'].isna() & cleaned_df['date'].notna()][['date', 'water_glasses']]

# Aggregate water logs
if not water_logs_all.empty:
    water_daily_totals = water_logs_all.groupby(water_logs_all['date'].dt.date)['water_glasses'].sum().reset_index()
    water_daily_totals.columns = ['date', 'total_water_glasses']
    water_daily_totals['date'] = pd.to_datetime(water_daily_totals['date'])
else:
    water_daily_totals = pd.DataFrame(columns=['date', 'total_water_glasses'])

# Calculate average cycle length (same as dashboard)
if not period_logs.empty:
    sorted_logs = period_logs.sort_values('date_start').reset_index(drop=True)
    sorted_logs['date_start'] = pd.to_datetime(sorted_logs['date_start'])
    sorted_logs['date_end'] = pd.to_datetime(sorted_logs['date_end'])
    cycle_lengths = []
    for i in range(1, len(sorted_logs)):
        cycle_len = (sorted_logs.iloc[i]['date_start'] - sorted_logs.iloc[i-1]['date_end']).days
        if cycle_len > 0:
            cycle_lengths.append(cycle_len)
    avg_cycle_length = round(np.mean(cycle_lengths), 1) if cycle_lengths else "N/A"
else:
    avg_cycle_length = "N/A"

# Other metrics
most_common_mood = (
    period_logs['mood'].mode()[0] if 'mood' in period_logs.columns and not period_logs['mood'].isna().all() else "N/A"
)

most_common_symptom = (
    period_logs['symptoms'].mode()[0] if 'symptoms' in period_logs.columns and not period_logs['symptoms'].isna().all() else "N/A"
)

total_logged_cycles = len(period_logs) if not period_logs.empty else 0
total_water_days = len(water_daily_totals) if not water_daily_totals.empty else 0
avg_daily_water = round(water_daily_totals['total_water_glasses'].mean(), 1) if not water_daily_totals.empty else "N/A"

# Create summary
summary_lines = [
    "--- Summary ---",
    f"Average Cycle Length: {avg_cycle_length} days",
    f"Most Common Mood: {most_common_mood}",
    f"Most Frequent Symptom: {most_common_symptom}",
    f"Total Logged Cycles: {total_logged_cycles}",
    f"Total Days with Water Tracking: {total_water_days}",
    f"Average Daily Water Intake: {avg_daily_water} glasses",
    f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
]

# Simulate CSV creation
buffer = io.StringIO()

buffer.write("=== PERIOD LOGS ===\n")
if not period_logs.empty:
    period_logs.to_csv(buffer, index=False)
else:
    buffer.write("No period logs available\n")

buffer.write("\n=== DAILY WATER INTAKE SUMMARY ===\n")
if not water_daily_totals.empty:
    water_daily_totals.to_csv(buffer, index=False)
else:
    buffer.write("No water logs available\n")

buffer.write("\n=== RAW WATER ENTRIES (Detailed) ===\n")
if not water_logs_all.empty:
    water_logs_all.to_csv(buffer, index=False)
else:
    buffer.write("No detailed water entries available\n")

buffer.write("\n\n" + "\n".join(summary_lines))

csv_content = buffer.getvalue()
print("CSV Content:")
print(csv_content)
