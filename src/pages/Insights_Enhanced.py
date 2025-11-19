import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

st.set_page_config(page_title="Insights & Export 🌸", layout="wide")

st.markdown("<h2 style='text-align:center; color:#D16BA5;'>Insights & Export 📊</h2>", unsafe_allow_html=True)
st.write("")

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

# Consistency Index
std_cycle = combined["Cycle_Length"].std()
consistency_score = max(0, 100 - (std_cycle / avg_cycle) * 100) if avg_cycle > 0 else 0

# --- Metrics Display ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Cycle Length", f"{avg_cycle:.1f} days")
col2.metric("Avg Period Length", f"{avg_period:.1f} days")
col3.metric("Cycle Variation", variation)
col4.metric("Most Common Symptom", common_symptom)

# --- Cycle Length Trend ---
st.subheader("📈 Cycle Length Trend")

# Convert date columns to datetime safely
combined["Start"] = pd.to_datetime(combined["Start"], errors="coerce")
combined["End"] = pd.to_datetime(combined["End"], errors="coerce")

# Calculate cycle length if missing
if "Cycle_Length" not in combined.columns:
    combined["Cycle_Length"] = (combined["End"] - combined["Start"]).dt.days

# Drop invalid or incomplete data
trend_df = combined.dropna(subset=["Start", "Cycle_Length"]).sort_values("Start")

if len(trend_df) > 1:
    # Create line chart
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot pink line with points
    ax.plot(
        trend_df["Start"],
        trend_df["Cycle_Length"],
        color="#F48FB1",
        marker="o",
        linewidth=2,
        label="Cycle Length"
    )

    # Add smooth shaded area below line
    ax.fill_between(
        trend_df["Start"],
        trend_df["Cycle_Length"],
        color="#FCE4EC",
        alpha=0.3
    )

    # Proper labels and scaling
    ax.set_title("Cycle Length Over Time", fontsize=13, color="#C2185B", pad=10)
    ax.set_xlabel("Cycle Start Date", fontsize=10)
    ax.set_ylabel("Cycle Length (days)", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    # Render chart
    st.pyplot(fig)

    # Smart trend analysis
    avg_diff = trend_df["Cycle_Length"].diff().mean()
    if avg_diff > 0.5:
        st.warning("Your cycles are gradually lengthening — monitor for consistency 💧")
    elif avg_diff < -0.5:
        st.success("Cycle lengths are shortening — trending toward regularity 🌷")
    else:
        st.info("Your cycle lengths are stable — great hormonal balance 🌸")

else:
    st.info("Not enough data points to show a trend yet. Log at least two complete cycles 🌼")

# --- Symptom Frequency ---
if not symptom_counts.empty:
    st.subheader("🌸 Symptom Frequency")
    fig, ax = plt.subplots()
    symptom_counts.plot(kind="bar", color="#F4A6C1", ax=ax)
    ax.set_ylabel("Occurrences")
    ax.set_xlabel("Symptom Type")
    st.pyplot(fig)

# --- Mood Distribution ---
st.subheader("🌙 Mood Distribution")

# Count mood occurrences
if "mood" in combined.columns:
    mood_series = combined["mood"].str.split(',').explode().str.strip()
    mood_counts = mood_series.value_counts()

    if not mood_counts.empty:
        # Define color palette for moods
        mood_colors = {
            "Happy": "#F48FB1",     # Pink
            "Neutral": "#CE93D8",   # Lavender
            "Sad": "#F8BBD0",       # Soft rose
            "Tired": "#E1BEE7",     # Lilac
            "Anxious": "#B39DDB",   # Light violet
            "Irritable": "#FFCDD2", # Blush red
            "Calm": "#D1C4E9",      # Pale lavender
        }

        # Map colors dynamically to available moods
        colors = [mood_colors.get(m, "#F48FB1") for m in mood_counts.index]

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            mood_counts,
            labels=mood_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': "black", 'fontsize': 9}
        )
        ax.axis('equal')
        st.pyplot(fig)

        # Add color legend
        legend_html = """
        <div style='margin-top:10px; display:flex; flex-wrap:wrap; gap:15px;'>
        """ + "".join(
            [
                f"<div style='display:flex; align-items:center; gap:5px; font-size:13px; color:#444;'>"
                f"<span style='width:15px; height:15px; border-radius:50%; background:{mood_colors.get(mood, '#F48FB1')}; display:inline-block;'></span>"
                f"{mood}</div>"
                for mood in mood_counts.index
            ]
        ) + "</div>"

        st.markdown(legend_html, unsafe_allow_html=True)

        st.caption("💡 Each color represents a recorded mood in your cycle history 🌸")
    else:
        st.info("No mood data available yet. Log moods to see your distribution 🌷")
else:
    st.warning("Mood data column not found in dataset.")

# --- Hydration Impact on Symptoms ---
if "water_intake" in combined.columns and not symptom_counts.empty:
    st.subheader("💧 Hydration Impact on Symptoms")
    # Group by individual symptoms
    combined_exploded = combined.assign(symptoms=combined["symptoms"].str.split(',')).explode("symptoms")
    combined_exploded["symptoms"] = combined_exploded["symptoms"].str.strip()
    hydration_symptom = combined_exploded.groupby("symptoms")["water_intake"].mean().sort_values()
    fig, ax = plt.subplots()
    hydration_symptom.plot(kind="barh", color="#E79EC5", ax=ax)
    ax.set_xlabel("Avg Water Intake (cups)")
    st.pyplot(fig)

# --- Mood Trend ---
if "mood" in combined.columns and not combined["mood"].dropna().empty:
    st.subheader("🌈 Mood Trend")
    mood_series = combined["mood"].str.split(',').explode().str.strip()
    mood_over_time = combined.assign(mood=mood_series).groupby(combined["Start"].dt.date)["mood"].value_counts().unstack().fillna(0)
    fig, ax = plt.subplots()
    for mood in mood_over_time.columns:
        ax.plot(mood_over_time.index, mood_over_time[mood], label=mood, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

# --- Consistency Index ---
st.subheader("⚖️ Cycle Consistency Index")
st.metric("Consistency Score", f"{consistency_score:.1f}%")

# --- Smart Insights ---
st.subheader("💬 Smart Insights")
if consistency_score > 85:
    st.success("Your cycles are highly regular — excellent hormonal balance 🌸")
elif consistency_score > 65:
    st.warning("Some minor variations detected — maintain hydration and rest 💧")
else:
    st.error("Irregular cycles observed — track closely and consult if needed 🩺")

# --- Summary Card ---
st.markdown(f"""
**🩷 Summary for You**
- Average cycle length: **{avg_cycle:.1f} days**
- Most common symptom: **{common_symptom}**
- Cycle consistency: **{consistency_score:.1f}%**
""")

# --- Export Section ---
st.subheader("📤 Export Data & Visuals")
summary = {
    "Average Cycle Length": avg_cycle,
    "Average Period Length": avg_period,
    "Cycle Variation": variation,
    "Consistency (%)": consistency_score,
    "Most Common Symptom": common_symptom
}
summary_df = pd.DataFrame([summary])

csv = summary_df.to_csv(index=False)
st.download_button("⬇️ Export Summary CSV", csv, "cycle_summary.csv", "text/csv")

buf = io.BytesIO()
fig.savefig(buf, format="png")
st.download_button("🖼️ Export Trend Chart", buf.getvalue(), "cycle_trend.png", "image/png")
