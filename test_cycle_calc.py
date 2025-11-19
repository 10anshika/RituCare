import pandas as pd
import numpy as np

# Load logs
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

avg_cycle = np.mean(cycle_lengths) if cycle_lengths else 28
print(f"Average cycle length: {avg_cycle:.1f} days")
print(f"Cycle lengths: {cycle_lengths}")
