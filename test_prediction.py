import pandas as pd
from datetime import timedelta

# Load datasets
personal_df = pd.read_csv('dataset/personal_cycle_logs_filled.csv')
logs_df = pd.read_csv('logs/user_cycle_log.csv')

# Convert to datetime
personal_df['Start_Date'] = pd.to_datetime(personal_df['Start_Date'], errors='coerce')
personal_df['End_Date'] = pd.to_datetime(personal_df['End_Date'], errors='coerce')

if 'date_start' in logs_df.columns:
    logs_df['date_start'] = pd.to_datetime(logs_df['date_start'], errors='coerce')
if 'date_end' in logs_df.columns:
    logs_df['date_end'] = pd.to_datetime(logs_df['date_end'], errors='coerce')

# Combine both datasets
combined = pd.concat([
    personal_df.rename(columns={'Start_Date': 'Start', 'End_Date': 'End'}),
    logs_df.rename(columns={'date_start': 'Start', 'date_end': 'End'})
], ignore_index=True).dropna(subset=['Start', 'End'])

# Sort by Start date
combined = combined.sort_values('Start').reset_index(drop=True)

print('Combined cycles:')
for i, row in combined.iterrows():
    print(f'{i+1}: {row["Start"].date()} - {row["End"].date()}')

# Calculate cycle lengths (days between consecutive starts)
cycle_lengths = []
for i in range(len(combined) - 1):
    cycle_len = (combined.iloc[i+1]['Start'] - combined.iloc[i]['Start']).days
    if cycle_len > 0:
        cycle_lengths.append(cycle_len)
        print(f'Cycle length {i+1}: {cycle_len} days')

print(f'Cycle lengths: {cycle_lengths}')
print(f'Average cycle length: {sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else "N/A"}')

# Get last cycle end date
last_end = combined['End'].max()
print(f'Last end date: {last_end.date()}')

# Predict next period
if cycle_lengths:
    user_avg = sum(cycle_lengths) / len(cycle_lengths)
    print(f'User average cycle: {user_avg}')
    next_start = last_end + timedelta(days=user_avg)
else:
    print('No cycle lengths available')
    next_start = last_end + timedelta(days=28)  # Default

next_end = next_start + timedelta(days=5)
print(f'Predicted next period: {next_start.date()} - {next_end.date()}')
