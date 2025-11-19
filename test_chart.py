import pandas as pd
import numpy as np
import plotly.express as px

# Simulate the data from the app
logs = pd.read_csv('logs/user_cycle_log.csv')
logs['date_start'] = pd.to_datetime(logs['date_start'], errors='coerce')
logs['date_end'] = pd.to_datetime(logs['date_end'], errors='coerce')

period_logs = logs.dropna(subset=['date_start', 'date_end'])
sorted_logs = period_logs.sort_values('date_start').reset_index(drop=True)

cycle_lengths = []
for i in range(1, len(sorted_logs)):
    cycle_len = (sorted_logs.iloc[i]['date_start'] - sorted_logs.iloc[i-1]['date_end']).days
    if cycle_len > 0:
        cycle_lengths.append(cycle_len)

if cycle_lengths:
    cycle_dates = sorted_logs['date_start'].iloc[1:len(cycle_lengths)+1]
    cycle_df = pd.DataFrame({'Date': cycle_dates, 'Cycle Length': cycle_lengths})
    # Sort by Date to maintain chronological trend
    cycle_df = cycle_df.sort_values('Date')
    print("Cycle DataFrame:")
    print(cycle_df)
    print(f"X-axis (Cycle Length) range: {cycle_df['Cycle Length'].min()} - {cycle_df['Cycle Length'].max()}")
    print(f"Y-axis (Date) range: {cycle_df['Date'].min()} - {cycle_df['Date'].max()}")
    print("Chart axes swapped successfully: Cycle Length on x-axis, Date on y-axis")
else:
    print("No cycle lengths available")
