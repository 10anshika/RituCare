import pandas as pd
from datetime import timedelta, datetime
import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    Prophet = None
    prophet_available = False

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


def forecast_cycle_length_arima(cycle_lengths, steps=1):
    """
    Forecast next cycle length using ARIMA model.
    """
    if ARIMA is None or len(cycle_lengths) < 3:
        return None

    try:
        # Reset index to avoid unsupported index warnings
        cycle_lengths = cycle_lengths.reset_index(drop=True)
        # Fit ARIMA model (order can be tuned)
        model = ARIMA(cycle_lengths, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=steps)
        return forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
    except Exception:
        return None


def get_next_period_prediction(personal_path="dataset/personal_cycle_logs_filled.csv",
                            logs_path="logs/user_cycle_log.csv"):
 """
 Adaptive next-period prediction using trend-based cycle modeling and ARIMA for precision.
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

 # Compute cycle lengths
 starts = combined["Start"]
 cycle_lengths = starts.diff().dt.days.dropna()
 cycle_lengths = cycle_lengths[(cycle_lengths >= 15) & (cycle_lengths <= 60)]

 if cycle_lengths.empty:
     # Fallback to default
     final_cycle = 28
     slope = 0
 else:
     # Use ARIMA for precise forecast if possible
     arima_forecast = forecast_cycle_length_arima(cycle_lengths, steps=1)
     if arima_forecast and 15 <= arima_forecast <= 60:
         final_cycle = round(arima_forecast, 1)
         slope = 0  # ARIMA handles trend internally
     else:
         # Fallback to adaptive
         adaptive_cycle, slope = compute_adaptive_cycle_length(combined)
         final_cycle = adaptive_cycle if adaptive_cycle else 28

 # Integrate fine-tuned model if available
 fine_tuned_cycle = None
 if fine_tuned_forecast:
     try:
         model_cycle = fine_tuned_forecast().get("fine_tuned_cycle_length", None)
         if model_cycle and 15 <= model_cycle <= 60:
             fine_tuned_cycle = float(model_cycle)
     except Exception:
         fine_tuned_cycle = None

 # Blend with fine-tuned if available
 if fine_tuned_cycle:
     final_cycle = round(0.7 * final_cycle + 0.3 * fine_tuned_cycle, 1)

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
 cycle_std = cycle_lengths.std() if not cycle_lengths.empty else None
 if cycle_std is None or np.isnan(cycle_std):
     confidence = "Low"
 elif cycle_std <= 2:
     confidence = "High"
 elif cycle_std <= 5:
     confidence = "Moderate"
 else:
     confidence = "Low"

 # Calculate possible range based on confidence and variation
 cycle_std = cycle_lengths.std() if not cycle_lengths.empty else 5  # default std if no data
 if confidence == "High":
     range_days = max(1, int(cycle_std * 0.5))  # smaller range for high confidence
 elif confidence == "Moderate":
     range_days = max(2, int(cycle_std * 0.75))
 else:
     range_days = max(3, int(cycle_std))  # larger range for low confidence

 possible_start_from = (last_start + timedelta(days=final_cycle - range_days)).date()
 possible_start_to = (last_start + timedelta(days=final_cycle + range_days)).date()

 return {
     "next_start": next_start,
     "next_end": next_end,
     "adaptive_cycle_length": final_cycle,
     "trend_slope": slope,
     "confidence": confidence,
     "possible_start_range": f"{possible_start_from} to {possible_start_to}"
 }


if __name__ == "__main__":
 print(get_next_period_prediction())
