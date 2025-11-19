import pandas as pd

# Load Data
personal_df = pd.read_csv("dataset/personal_cycle_logs_filled.csv")
user_logs = pd.read_csv("logs/user_cycle_log.csv")

# Convert to datetime
personal_df["Start_Date"] = pd.to_datetime(personal_df["Start_Date"], errors="coerce")
personal_df["End_Date"] = pd.to_datetime(personal_df["End_Date"], errors="coerce")
if "date_start" in user_logs.columns:
    user_logs["date_start"] = pd.to_datetime(user_logs["date_start"], errors="coerce")
if "date_end" in user_logs.columns:
    user_logs["date_end"] = pd.to_datetime(user_logs["date_end"], errors="coerce")

# Combine datasets
combined = pd.concat([
    personal_df.rename(columns={"Start_Date": "Start", "End_Date": "End", "Symptoms": "symptoms", "Mood": "mood", "Water_Intake": "water_intake"}),
    user_logs.rename(columns={"date_start": "Start", "date_end": "End", "water_glasses": "water_intake"})
], ignore_index=True).dropna(subset=["Start", "End"])

# Ensure datetime conversion
combined["Start"] = pd.to_datetime(combined["Start"], errors="coerce")
combined["End"] = pd.to_datetime(combined["End"], errors="coerce")

# Cycle Length
combined["Cycle_Length"] = (combined["End"] - combined["Start"]).dt.days
avg_cycle = combined["Cycle_Length"].mean()
avg_period = combined["Cycle_Length"].mean()  # Calculate actual average period length
variation = "Low" if combined["Cycle_Length"].std() < 3 else "Moderate" if combined["Cycle_Length"].std() < 6 else "High"

# Most Common Symptom
if "symptoms" in combined.columns and not combined["symptoms"].dropna().empty:
    symptom_series = combined["symptoms"].str.split(',').explode().str.strip()
    symptom_counts = symptom_series.value_counts()
    common_symptom = symptom_counts.idxmax()
else:
    common_symptom = "N/A"
    symptom_counts = pd.Series()

print("Avg cycle length:", avg_cycle)
print("Variation:", variation)
print("Common symptom:", common_symptom)
print("Combined shape:", combined.shape)
