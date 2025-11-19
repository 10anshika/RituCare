import pandas as pd
from datetime import timedelta, datetime
import numpy as np

try:
 from ritucare.core.fine_tuning_model import fine_tuned_forecast
except Exception:
 fine_tuned_forecast = None


def compute_adaptive_cycle_length(df, start_col="Start"):
 """
 Compute adaptive cycle lengths considering recent trends and variation.
 Returns a weighted adaptive average of cycle lengths.
 """
 starts = pd.to_datetime(df[start_col], errors="coerce").dropna().sort_values()
 if len(starts) < 2:
     return None

 # Calculate cycle lengths between consecutive period starts
 cycle_lengths = starts.diff().dt.days.dropna()

 # Remove implausible outliers
 cycle_lengths = cycle_lengths[(cycle_lengths >= 15) & (cycle_lengths <= 60)]
 if cycle_lengths.empty:
     return None

 # Weighted average favoring recent cycles (more recent = higher weight)
 weights = np.linspace(0.5, 1.5, len(cycle_lengths))
 weighted_avg = np.average(cycle_lengths, weights=weights)

 # Adjust adaptively based on slope (trend)
 x = np.arange(len(cycle_lengths))
 coeffs = np.polyfit(x, cycle_lengths, 1)  # linear regression slope
 slope = coeffs[0]

 adaptive_cycle = weighted_avg + slope * 0.5  # small adjustment factor

 # Smooth to avoid overreaction
 adaptive_cycle = max(21, min(40, adaptive_cycle))

 return round(adaptive_cycle, 1), round(slope, 3)


def get_next_period_prediction(personal_path="ritucare/data/personal_cycle_logs_filled.csv",
                            logs_path="ritucare/logs/user_cycle_log.csv"):
 """
 Adaptive next-period prediction using trend-based cycle modeling.
 """
 dfs = []
 for path in [personal_path, logs_path]:
     try:
         df = pd.read_csv(path)
         df = df.rename(columns={
             "Start_Date": "Start", "End_Date": "End",
             "date_start": "Start", "date_end": "End"
         })
         dfs.append(df)
     except Exception:
         continue

 if not dfs:
     raise FileNotFoundError("No valid cycle data found.")

 combined = pd.concat(dfs, ignore_index=True)
 combined["Start"] = pd.to_datetime(combined["Start"], errors="coerce")
 combined["End"] = pd.to_datetime(combined["End"], errors="coerce")

 combined = combined.dropna(subset=["Start"]).sort_values("Start")

 # Compute adaptive cycle length
 adaptive_cycle, slope = compute_adaptive_cycle_length(combined)

 # Integrate fine-tuned model if available
 fine_tuned_cycle = None
 if fine_tuned_forecast:
     try:
         model_cycle = fine_tuned_forecast().get("fine_tuned_cycle_length", None)
         if model_cycle and 15 <= model_cycle <= 60:
             fine_tuned_cycle = float(model_cycle)
     except Exception:
         fine_tuned_cycle = None

 # Blend adaptive and fine-tuned cycles
 if fine_tuned_cycle:
     final_cycle = round(0.8 * adaptive_cycle + 0.2 * fine_tuned_cycle, 1)
 else:
     final_cycle = adaptive_cycle

 # Determine last start and calculate prediction
 last_start = combined["Start"].iloc[-1]
 next_start = (last_start + timedelta(days=final_cycle)).date()

 # Estimate period length from historical averages
 if "End" in combined.columns and not combined["End"].dropna().empty:
     period_lengths = (combined["End"] - combined["Start"]).dt.days.dropna()
     avg_period_length = int(period_lengths.mean()) if not period_lengths.empty else 5
 else:
     avg_period_length = 5

 next_end = (pd.to_datetime(next_start) + timedelta(days=avg_period_length)).date()

 # Confidence scoring based on variation
 cycle_std = combined["Start"].diff().dt.days.std()
 if cycle_std is None or np.isnan(cycle_std):
     confidence = "Low"
 elif cycle_std <= 2:
     confidence = "High"
 elif cycle_std <= 5:
     confidence = "Moderate"
 else:
     confidence = "Low"

 return {
     "next_start": next_start,
     "next_end": next_end,
     "adaptive_cycle_length": final_cycle,
     "trend_slope": slope,
     "confidence": confidence
 }


if __name__ == "__main__":
 print(get_next_period_prediction())
