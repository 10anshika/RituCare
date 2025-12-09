import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pickle
import os
import sys
import warnings
import io
import plotly.express as px
warnings.filterwarnings('ignore')

# Add ritucare directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'ritucare'))

# Page config must be first
st.set_page_config(page_title="RituCare 🌸", page_icon="🌸", layout="wide")

# Enhanced CSS for modern wellness UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    body {
        background: linear-gradient(135deg, #fef7ff, #f8e8ff, #fce4ff, #f0e6ff, #fce8ff);
        background-attachment: fixed;
        animation: fadeIn 0.8s ease-in-out;
        margin: 0;
        padding: 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .ritucard {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 12px 40px rgba(255, 182, 193, 0.15), 0 4px 16px rgba(255, 105, 180, 0.1);
        border: 1px solid rgba(255, 192, 203, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .ritucard:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 48px rgba(255, 182, 193, 0.2), 0 8px 24px rgba(255, 105, 180, 0.15);
    }

    .header {
        text-align: center;
        background: linear-gradient(135deg, #ff9ecd, #d9a3ff, #ffb3d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Poppins', sans-serif;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 24px;
        position: relative;
    }

    .header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 3px;
        background: linear-gradient(90deg, #ff9ecd, #d9a3ff, #ffb3d9);
        border-radius: 2px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #6b46c1;
        margin-bottom: 16px;
    }

    .section-header {
        text-align: center;
        background: linear-gradient(135deg, #ff9ecd, #d9a3ff, #ffb3d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Poppins', sans-serif;
        font-size: 1.8em;
        font-weight: 600;
        margin: 32px 0 20px 0;
        position: relative;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: -6px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #ff9ecd, #d9a3ff, #ffb3d9);
        border-radius: 1px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 182, 193, 0.1), rgba(217, 163, 255, 0.1));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 192, 203, 0.2);
        border-radius: 0 24px 24px 0;
        padding: 24px 16px;
    }

    [data-testid="stSidebar"] .stRadio {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(255, 182, 193, 0.1);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #ff9ecd, #d9a3ff);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(255, 156, 205, 0.3);
        text-transform: none;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255, 156, 205, 0.4);
        background: linear-gradient(135deg, #ff8ac7, #c78eff);
    }

    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(255, 156, 205, 0.3);
    }

    /* Progress Bar Styling */
    .progress-bar {
        background: rgba(255, 182, 193, 0.2);
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .progress-fill {
        background: linear-gradient(90deg, #ff69b4, #ffb3d9, #d9a3ff);
        height: 100%;
        border-radius: 12px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* Form Styling */
    .stForm {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.15);
        border: 1px solid rgba(255, 192, 203, 0.2);
    }

    /* Input Styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stDateInput>div>input {
        border-radius: 12px;
        border: 2px solid rgba(255, 182, 193, 0.3);
        padding: 12px 16px;
        font-family: 'Inter', sans-serif;
        font-size: 1em;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stDateInput>div>input:focus {
        border-color: #ff9ecd;
        box-shadow: 0 0 0 3px rgba(255, 156, 205, 0.1);
        outline: none;
    }

    /* Checkbox Styling */
    .stCheckbox {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 8px 16px;
        margin: 4px 0;
        transition: all 0.3s ease;
    }

    .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: translateX(4px);
    }

    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(255, 182, 193, 0.1);
    }

    .stDataFrame table {
        border-radius: 16px;
    }

    .stDataFrame thead th {
        background: linear-gradient(135deg, #ff9ecd, #d9a3ff);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 12px;
        text-align: center;
    }

    .stDataFrame tbody td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid rgba(255, 182, 193, 0.1);
    }

    .stDataFrame tbody tr:nth-child(even) {
        background: rgba(255, 182, 193, 0.05);
    }

    .stDataFrame tbody tr:hover {
        background: rgba(255, 182, 193, 0.1);
        transform: scale(1.01);
        transition: all 0.2s ease;
    }

    /* Chart Styling */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(255, 182, 193, 0.1);
        background: rgba(255, 255, 255, 0.9);
    }

    /* Success/Warning/Error Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 16px;
        padding: 16px 20px;
        margin: 12px 0;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }

    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
    }

    .stInfo {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        color: #0c5460;
    }

    .stWarning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8b5cf6;
        font-size: 0.9em;
        margin-top: 40px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 20px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }

    /* Emoji Styling */
    .emoji {
        font-size: 1.5em;
        display: inline-block;
        margin-right: 8px;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .header {
            font-size: 2.2em;
        }

        .ritucard {
            padding: 16px;
            margin: 8px 0;
        }

        .stButton>button {
            padding: 10px 20px;
            font-size: 1em;
        }
    }

    /* Animation for page transitions */
    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 182, 193, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff9ecd, #d9a3ff);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff8ac7, #c78eff);
    }
</style>
""", unsafe_allow_html=True)

# Load helpers
from ritucare.core.phase_logic import get_phase, get_cycle_day as get_cycle_day_func
from ritucare.core.cycle_forecasting import forecast_next_period
import importlib
import ritucare.core.personal_forecast_manager as pfm

try:
    from ritucare.core.nutrition_engine import get_nutrition
except:
    def get_nutrition(phase): return f"Focus on iron-rich foods 💪"

try:
    from ritucare.core.pcos_model import predict_pcos_risk
except:
    def predict_pcos_risk(features): return {'error': 'Model not loaded'}

# Load models
cycle_model = None
pcos_model = None
try:
    with open('models/cycle_model.pkl', 'rb') as f:
        cycle_model = pickle.load(f)
except:
    pass

try:
    with open('models/pcos_model.pkl', 'rb') as f:
        pcos_model = pickle.load(f)
except:
    pass

# Load user logs
def load_logs():
    if os.path.exists('logs/user_cycle_log.csv'):
        df = pd.read_csv('logs/user_cycle_log.csv')
        df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
        df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')
        # Ensure all required columns exist
        required_columns = ['user_id', 'date_start', 'date_end', 'flow', 'mood', 'symptoms', 'notes', 'water_glasses']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        # Add date column for water entries if not present
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    return pd.DataFrame(columns=['user_id', 'date_start', 'date_end', 'flow', 'mood', 'symptoms', 'notes', 'water_glasses', 'date'])

def save_logs(df):
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/user_cycle_log.csv', index=False)

logs = load_logs()

# Sidebar navigation
st.sidebar.title("🌸 RituCare Navigation")
page = st.sidebar.radio("Navigation", ["🌸 Dashboard", "📝 Log Period", "💧 Water Tracker", "❤️ PCOS Check", "📊 Insights & Export"], label_visibility="hidden")

# Cycle day indicator
def get_cycle_day():
    if logs.empty:
        return None, "Log your first period to start tracking! 🌸"
    
    # Filter out NaN date_start values (water tracker entries)
    valid_dates = logs['date_start'].dropna()
    if valid_dates.empty:
        return None, "Log your first period to start tracking! 🌸"
    
    last_start = pd.to_datetime(valid_dates.max()).date()
    
    # Calculate user's average cycle length
    user_cycle_length = 28  # Default
    if len(valid_dates) >= 2:
        sorted_dates = pd.to_datetime(valid_dates).sort_values()
        cycle_lengths = sorted_dates.diff().dt.days.dropna()
        # Filter to reasonable range
        cycle_lengths = cycle_lengths[(cycle_lengths >= 15) & (cycle_lengths <= 60)]
        if not cycle_lengths.empty:
            user_cycle_length = int(cycle_lengths.mean())
    
    # Calculate cycle day with personalized length
    cycle_day = get_cycle_day_func(last_start, cycle_length=user_cycle_length)
    
    if cycle_day:
        if user_cycle_length != 28:
            return cycle_day, f"Today is Day {cycle_day} of your {user_cycle_length}-day cycle 🌸"
        else:
            return cycle_day, f"Today is Day {cycle_day} of your cycle 🌸"
    else:
        return None, "Cycle complete! Log next period start. 💕"


if page == "🌸 Dashboard":
    cycle_day, cycle_msg = get_cycle_day()
    st.markdown('<div class="header">Welcome to RituCare 🌸</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><h3>{cycle_msg}</h3></div>', unsafe_allow_html=True)

    # Greeting
    st.markdown('<div class="glass-card"><h4>Good day, beautiful soul! 💕</h4><p>Your body is amazing. Keep listening to it.</p></div>', unsafe_allow_html=True)

    # Card grid
    col1, col2, col3 = st.columns(3)
    with col1:
        phase_full = get_phase(cycle_day) if cycle_day else "Unknown"
        phase_name = phase_full.split()[0] if phase_full != "Unknown" else "Unknown"

        # --- Nutrition Recommendations based on Phase ---
        if phase_name == "Menstrual":
            nutrition_tip = """
            💧 Stay hydrated and eat **iron-rich foods** like spinach, dates, and lentils.  
            🍫 A bit of **dark chocolate** helps replenish magnesium and improve mood.  
            🫐 Try **warm soups, berries, and leafy greens** to ease cramps and fight fatigue.
            """
        elif phase_name == "Follicular":
            nutrition_tip = """
            🥑 Boost energy with **healthy fats and lean proteins** — avocado, eggs, salmon.  
            🍊 Add **vitamin C foods** (oranges, amla) to support rising estrogen.  
            🌱 Fresh veggies and sprouts help your body rebuild after menstruation.
            """
        elif phase_name == "Ovulatory" or phase_name == "Ovulation":
            nutrition_tip = """
            🍓 Eat **antioxidant-rich foods** (berries, nuts, seeds) to support ovulation.  
            🥦 Include **zinc and B vitamins** from chickpeas, pumpkin seeds, and whole grains.  
            💕 Stay hydrated — this phase raises your metabolism slightly!
            """
        elif phase_name == "Luteal":
            nutrition_tip = """
            🍠 Balance mood swings with **complex carbs** like sweet potatoes and oats.  
            🧘‍♀️ Add **magnesium-rich foods** (bananas, dark chocolate, almonds) for calmness.  
            ☕ Avoid excess caffeine — herbal teas work wonders in this phase.
            """
        else:
            nutrition_tip = "🌸 Eat balanced meals, stay hydrated, and listen to your body’s needs."

    # Display the phase and nutrition advice
        with st.container():
            st.markdown(
                f"""
                <div class='ritucard'>
                <h4 style='color:#d63384;'>Current Phase: {phase_name}</h4>
                <div style='margin-top:8px; line-height:1.6;'>{nutrition_tip}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with col2:
        # Force reload so Streamlit doesn’t use cached logic
        importlib.reload(pfm)

        # Clear old session cache if present
        if "forecast" in st.session_state:
            del st.session_state["forecast"]

        # Run updated forecast function
        try:
            prediction = pfm.get_next_period_prediction()
            next_period = f"{prediction['next_start']} - {prediction['next_end']}<br><small>Possible start: {prediction['possible_start_range']}</small>"
        except Exception as e:
            next_period = "Log to predict"
        st.markdown(f'<div class="glass-card"><h4>📅 Next Period</h4><p>{next_period}</p></div>', unsafe_allow_html=True)
    with col3:
        today = datetime.now().date()
        water_logs = logs[pd.isna(logs['date_start']) & pd.notna(logs['date']) & (logs['date'].dt.date == today)] if not logs.empty else pd.DataFrame()
        water_today = water_logs['water_glasses'].sum() if not water_logs.empty else 0
        goal = 10  # Default goal for dashboard
        progress = min(water_today / goal * 100, 100)
        st.markdown(f'<div class="glass-card"><h4>💧 Hydration</h4><p>{water_today}/{goal} cups</p><div class="progress-bar"><div class="progress-fill" style="width:{progress}%"></div></div></div>', unsafe_allow_html=True)

    # Cycle progress
    if cycle_day:
        # Calculate user's average cycle length (same logic as above)
        user_cycle_length = 28
        valid_dates = logs['date_start'].dropna()
        if len(valid_dates) >= 2:
            sorted_dates = pd.to_datetime(valid_dates).sort_values()
            cycle_lengths = sorted_dates.diff().dt.days.dropna()
            cycle_lengths = cycle_lengths[(cycle_lengths >= 15) & (cycle_lengths <= 60)]
            if not cycle_lengths.empty:
                user_cycle_length = int(cycle_lengths.mean())
        
        progress = (cycle_day / user_cycle_length) * 100
        st.markdown(f'<div class="glass-card"><h4>Cycle Progress</h4><div class="progress-bar"><div class="progress-fill" style="width:{progress}%"></div></div><p>Day {cycle_day}/{user_cycle_length}</p></div>', unsafe_allow_html=True)

elif page == "📝 Log Period":
    st.markdown('<div class="header">Log Your Period 📝</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><p>Log your period details and symptoms to help track cycle patterns 🌷</p></div>', unsafe_allow_html=True)
    with st.form("log_form"):
        date_start = st.date_input("Period Start Date")
        date_end = st.date_input("Period End Date", value=date_start + timedelta(days=5))
        flow = st.slider("Flow Intensity", 1, 5)
        mood = st.multiselect("Mood", ["Happy", "Sad", "Irritable", "Calm"])
        st.markdown("### Symptoms")
        cramps = st.checkbox("Cramps 💢")
        acne = st.checkbox("Acne ⚪")
        bloating = st.checkbox("Bloating 🎈")
        fatigue = st.checkbox("Fatigue 💤")
        back_pain = st.checkbox("Back Pain 🧍‍♀️💥")
        breast_tenderness = st.checkbox("Breast Tenderness 🤕")
        notes = st.text_area("Notes (optional)", placeholder="Any additional notes...")
        submitted = st.form_submit_button("Log Period 💕")
        if submitted:
            selected_symptoms = []
            if cramps: selected_symptoms.append("Cramps")
            if acne: selected_symptoms.append("Acne")
            if bloating: selected_symptoms.append("Bloating")
            if fatigue: selected_symptoms.append("Fatigue")
            if back_pain: selected_symptoms.append("Back Pain")
            if breast_tenderness: selected_symptoms.append("Breast Tenderness")
            symptoms_str = ', '.join(selected_symptoms)
            new_log = pd.DataFrame([{
                'user_id': 'user1',
                'date_start': pd.Timestamp(date_start),
                'date_end': pd.Timestamp(date_end),
                'flow': flow,
                'mood': ', '.join(mood),
                'symptoms': symptoms_str,
                'notes': notes
            }])
            logs = pd.concat([logs, new_log], ignore_index=True)
            save_logs(logs)
            st.success("Period logged successfully! 🌸 Keep glowing!")

    # Update End Date for Current Period
    st.markdown('<div class="glass-card"><h4>Update End Date for Current Period</h4><p>If your period is ongoing, update the end date here once it ends.</p></div>', unsafe_allow_html=True)
    if not logs.empty:
        last_period = logs.dropna(subset=['date_start']).sort_values('date_start', ascending=False).iloc[0] if not logs.dropna(subset=['date_start']).empty else None
        if last_period is not None:
            with st.form("update_end_date_form"):
                new_end_date = st.date_input("New End Date", value=pd.to_datetime(last_period['date_end']).date() if pd.notna(last_period['date_end']) else None)
                update_submitted = st.form_submit_button("Update End Date 💕")
                if update_submitted:
                    logs.loc[logs['date_start'] == last_period['date_start'], 'date_end'] = pd.Timestamp(new_end_date)
                    save_logs(logs)
                    st.success("End date updated successfully! 🌸")

    # Recent Cycles History
    st.markdown('<div class="glass-card"><h4>Recent Cycles History</h4></div>', unsafe_allow_html=True)
    recent_cycles = logs.dropna(subset=['date_start']).sort_values('date_start', ascending=False).head(5)
    if not recent_cycles.empty:
        # Select only existing columns to avoid KeyError if CSV has old structure
        available_cols = ['date_start', 'date_end', 'flow', 'mood', 'symptoms', 'notes']
        existing_cols = [col for col in available_cols if col in recent_cycles.columns]
        # Add serial number column
        recent_cycles_display = recent_cycles[existing_cols].reset_index(drop=True)
        recent_cycles_display.insert(0, 'S.No.', range(1, len(recent_cycles_display) + 1))
        # Format date columns to show only date without time
        if 'date_start' in recent_cycles_display.columns:
            recent_cycles_display['date_start'] = recent_cycles_display['date_start'].dt.date
        if 'date_end' in recent_cycles_display.columns:
            recent_cycles_display['date_end'] = recent_cycles_display['date_end'].dt.date
        st.dataframe(recent_cycles_display)
    else:
        st.info("No cycles logged yet.")

    # Delete Specific Cycle
    st.markdown('<div class="glass-card"><h4>Delete Specific Cycle</h4><p>Select a cycle to delete from your recent history.</p></div>', unsafe_allow_html=True)
    if not recent_cycles.empty:
        cycle_options = recent_cycles['date_start'].dt.strftime('%Y-%m-%d').tolist()
        selected_cycle = st.selectbox("Select Cycle Start Date to Delete", cycle_options, key="delete_cycle")
        if st.button("Delete Selected Cycle 💔"):
            selected_date = pd.to_datetime(selected_cycle)
            logs = logs[logs['date_start'] != selected_date]
            save_logs(logs)
            st.success("Cycle deleted successfully! 🌸")
            st.rerun()
    else:
        st.info("No cycles available to delete.")

elif page == "💧 Water Tracker":
    st.markdown('<div class="header">Water Tracker 💧</div>', unsafe_allow_html=True)

    # Load or set default goal
    goal_key = "water_goal"
    if goal_key not in st.session_state:
        st.session_state[goal_key] = 10  # Default 10 glasses

    # Goal customization
    st.markdown('<div class="glass-card"><h4>Customize Your Daily Goal</h4></div>', unsafe_allow_html=True)
    new_goal = st.slider("Daily Water Goal (glasses)", min_value=1, max_value=20, value=st.session_state[goal_key], step=1)
    if new_goal != st.session_state[goal_key]:
        st.session_state[goal_key] = new_goal
        st.success("Goal updated! 💧")

    # Current progress
    today = datetime.now().date()
    water_logs = logs[pd.isna(logs['date_start']) & pd.notna(logs['date']) & (logs['date'].dt.date == today)] if not logs.empty else pd.DataFrame()
    current_glasses = int(water_logs['water_glasses'].sum()) if not water_logs.empty else 0
    goal = st.session_state[goal_key]
    progress = min(current_glasses / goal * 100, 100)

    st.markdown('<div class="glass-card"><h4>Today\'s Progress</h4></div>', unsafe_allow_html=True)
    st.write(f"**{current_glasses} / {goal} glasses**")
    st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width:{progress}%"></div></div>', unsafe_allow_html=True)

    # Add/Remove water
    st.markdown('<div class="glass-card"><h4>Log Water Intake</h4></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add 1 Glass 💧"):
            new_log = pd.DataFrame([{
                'date': pd.Timestamp(today),
                'water_glasses': 1.0
            }])
            logs = pd.concat([logs, new_log], ignore_index=True)
            save_logs(logs)
            st.rerun()
    with col2:
        if st.button("➖ Remove 1 Glass ❌") and current_glasses > 0:
            # Remove the last added glass for today
            today_entries = logs[(logs['date'].dt.date == today) & pd.notna(logs['water_glasses'])]
            if not today_entries.empty:
                last_entry_idx = today_entries.index[-1]
                logs = logs.drop(last_entry_idx)
                save_logs(logs)
                st.rerun()

    # Reset today's water
    if st.button("🔄 Reset Today 💧"):
        # Remove all water entries for today
        logs = logs[~((logs['date'].dt.date == today) & pd.notna(logs['water_glasses']))]
        save_logs(logs)
        st.success("Today's water log reset! 💧")
        st.rerun()

    # Recent history
    st.markdown('<div class="glass-card"><h4>Recent Water History</h4></div>', unsafe_allow_html=True)
    water_logs_all = logs[pd.isna(logs['date_start']) & pd.notna(logs['date'])]
    if not water_logs_all.empty:
        # Aggregate by date to show total glasses per day
        water_history = water_logs_all.groupby(water_logs_all['date'].dt.date)['water_glasses'].sum().reset_index()
        water_history.columns = ['date', 'total_glasses']
        water_history = water_history.sort_values('date', ascending=False).head(10)
        st.dataframe(water_history, use_container_width=True)
    else:
        st.info("No water logs yet. Start tracking! 💧")

elif page == "❤️ PCOS Check":
    st.markdown('<div class="header">PCOS Wellness Check 🎗️🌸</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><p><strong>Disclaimer:</strong> This is not medical advice. Consult a healthcare professional for diagnosis.</p></div>', unsafe_allow_html=True)

    with st.form("pcos_form"):
        # Basic Info
        st.markdown("### Basic Information")
        age = st.number_input("Age", 10, 60, value=25)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, value=60.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, value=160.0)
        bmi = weight / ((height/100)**2) if height > 0 else 0
        st.write(f"Calculated BMI: {bmi:.1f}")

        cycle_length = st.number_input("Cycle Length (days)", 20, 45, value=28)

        # Symptoms Section
        st.markdown("### Symptom Check 🌸")
        st.markdown("Select any symptoms you've experienced recently:")

        irregular_cycles = st.checkbox("Irregular cycles (>35 days)", help="Cycles longer than 35 days or unpredictable timing.")
        severe_acne = st.checkbox("Severe acne", help="Persistent or severe acne, especially on face, chest, or back.")
        excess_hair = st.checkbox("Excess facial/body hair", help="Unwanted hair growth in areas like face, chest, or abdomen.")
        hair_thinning = st.checkbox("Hair thinning / hair fall", help="Noticeable hair loss or thinning on scalp.")
        dark_patches = st.checkbox("Dark patches (neck/armpits)", help="Darkened skin patches, often called acanthosis nigricans.")
        weight_gain = st.checkbox("Sudden weight gain / hard to lose weight", help="Difficulty losing weight or unexplained weight gain.")
        sugar_cravings = st.checkbox("Sugar cravings / binge eating", help="Frequent cravings for sweets or episodes of overeating.")
        pelvic_pain = st.checkbox("Pelvic pain or ovarian cyst symptoms", help="Pain in pelvic area or symptoms related to ovarian cysts.")

        # Count symptoms
        symptoms = [irregular_cycles, severe_acne, excess_hair, hair_thinning, dark_patches, weight_gain, sugar_cravings, pelvic_pain]
        symptom_count = sum(symptoms)

        # Advanced Lab Values (optional)
        with st.expander("If you have lab values, enter below (optional)"):
            lh_level = st.number_input("LH Level (mIU/mL)", 0.0, 100.0, value=0.0)
            fsh_level = st.number_input("FSH Level (mIU/mL)", 0.0, 100.0, value=0.0)
            testosterone = st.number_input("Testosterone (ng/dL)", 0.0, 200.0, value=0.0)

        submitted = st.form_submit_button("Check Wellness 💕")

        if submitted:
            # Symptom-based logic
            symptom_risk = "Higher PCOS likelihood" if symptom_count >= 3 else "Low symptom indicators"

            # ML Model Prediction
            features = {
                'LengthofCycle': cycle_length,
                'BMI': bmi,
                'Age': age,
                'Weight': weight,
                'Height': height,
                'UnusualBleeding': int(irregular_cycles),
                'ReproductiveCategory': 0,  # Default, since removed
                'Maristatus': 0,  # Default
                'Numberpreg': 0  # Default
            }

            result = predict_pcos_risk(features)

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                ml_risk_level = result['risk_level']
                probability = result['probability'] * 100

                # Combine symptom and ML
                if symptom_count >= 3 and result['prediction'] == 1:
                    overall_risk = "Elevated"
                    color = "warning"
                    message = "Your symptoms and patterns suggest possible hormonal imbalance. Not medical advice — consider consulting a gynecologist ❤️"
                elif symptom_count >= 3 or result['prediction'] == 1:
                    overall_risk = "Moderate"
                    color = "info"
                    message = "Some indicators suggest monitoring your wellness. Keep listening to your body and consult a professional if needed 💕"
                else:
                    overall_risk = "Low"
                    color = "success"
                    message = "Your inputs show low indicators. Continue nurturing your wellness routine 🌸"

                # Display result
                if color == "success":
                    st.success(f"💕 **{overall_risk}** ({probability:.1f}% ML probability, {symptom_count} symptoms)")
                elif color == "info":
                    st.info(f"💕 **{overall_risk}** ({probability:.1f}% ML probability, {symptom_count} symptoms)")
                else:
                    st.warning(f"💕 **{overall_risk}** ({probability:.1f}% ML probability, {symptom_count} symptoms)")

                st.markdown(f'<div class="glass-card"><p>{message}</p><p>💡 <a href="#" style="color:#d9b3ff;">Explore nutrition tips for hormonal balance</a></p></div>', unsafe_allow_html=True)

elif page == "📊 Insights & Export":
    st.markdown('<div class="header">Insights & Export 📊</div>', unsafe_allow_html=True)

    if logs.empty:
        st.info("No data logged yet. Start tracking to see insights! 🌸")
    else:
        period_logs = logs.dropna(subset=['date_start', 'date_end'])
        if period_logs.empty:
            st.info("No complete period logs yet. Log your periods to unlock insights! 🌸")
        else:
            # Calculate insights
            sorted_logs = period_logs.sort_values('date_start').reset_index(drop=True)
            sorted_logs['date_start'] = pd.to_datetime(sorted_logs['date_start'])
            sorted_logs['date_end'] = pd.to_datetime(sorted_logs['date_end'])
            cycle_lengths = []
            for i in range(1, len(sorted_logs)):
                cycle_len = (sorted_logs.iloc[i]['date_start'] - sorted_logs.iloc[i-1]['date_end']).days
                if cycle_len > 0:  # Ensure positive cycle length
                    cycle_lengths.append(cycle_len)
            avg_cycle = np.mean(cycle_lengths) if cycle_lengths else 28  # Default to 28 if no data
            period_lengths = (sorted_logs['date_end'] - sorted_logs['date_start']).dt.days
            avg_period = period_lengths.mean()
            std_cycle = np.std(cycle_lengths) if cycle_lengths else 0
            if std_cycle < 3:
                variation = "Low"
            elif std_cycle < 7:
                variation = "Medium"
            else:
                variation = "High"

            # Most common symptom
            all_symptoms = []
            if 'symptoms' in sorted_logs.columns:
                for sym in sorted_logs['symptoms'].dropna():
                    all_symptoms.extend([s.strip() for s in sym.split(',') if s.strip()])
            if all_symptoms:
                symptom_counts = pd.Series(all_symptoms).value_counts()
                most_common = symptom_counts.idxmax()
                symptom_emoji_map = {
                    "Cramps": "💢",
                    "Acne": "⚪",
                    "Bloating": "🎈",
                    "Fatigue": "💤",
                    "Back Pain": "🧍‍♀️💥",
                    "Breast Tenderness": "🤕"
                }
                symptom_emoji = symptom_emoji_map.get(most_common, "🌸")
                most_common_display = f"{symptom_emoji} {most_common}"
            else:
                most_common_display = "None"

            # Insight Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="glass-card"><h4>Avg Cycle Length</h4><p>{avg_cycle:.1f} days</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="glass-card"><h4>Avg Period Length</h4><p>{avg_period:.1f} days</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="glass-card"><h4>Cycle Variation</h4><p>{variation}</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="glass-card"><h4>Most Common Symptom</h4><p>{most_common_display}</p></div>', unsafe_allow_html=True)

            # Charts
            st.markdown('<div class="glass-card"><h4>Cycle Length Trend</h4></div>', unsafe_allow_html=True)
            if cycle_lengths:
                cycle_dates = sorted_logs['date_start'].iloc[1:len(cycle_lengths)+1]
                cycle_df = pd.DataFrame({'Date': cycle_dates, 'Cycle Length': cycle_lengths})
                # Sort by Date to maintain chronological trend
                cycle_df = cycle_df.sort_values('Date')
                fig = px.line(cycle_df, x='Cycle Length', y='Date', title='', markers=True, color_discrete_sequence=["#ffb6c1"])
                fig.update_layout(xaxis_title='Cycle Length (days)', yaxis_title='Date')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="glass-card"><h4>Symptom Frequency</h4></div>', unsafe_allow_html=True)
            if all_symptoms:
                symptom_df = pd.DataFrame({'Symptom': symptom_counts.index, 'Count': symptom_counts.values})
                # Add emojis to labels
                symptom_df['Symptom'] = symptom_df['Symptom'].map(lambda x: f"{symptom_emoji_map.get(x, '🌸')} {x}")
                st.bar_chart(symptom_df.set_index('Symptom'), color="#dda0dd")

            st.markdown('<div class="glass-card"><h4>Mood Distribution</h4></div>', unsafe_allow_html=True)
            all_moods = []
            for mood in sorted_logs['mood'].dropna():
                all_moods.extend([m.strip() for m in mood.split(',') if m.strip()])
            if all_moods:
                mood_counts = pd.Series(all_moods).value_counts()
                import plotly.express as px
                fig = px.pie(mood_counts, values=mood_counts.values, names=mood_counts.index, color_discrete_sequence=["#ffb6c1", "#dda0dd", "#f7c6e8", "#d9b3ff"])
                fig.update_traces(textinfo='label+percent')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Define color palette for moods
                mood_colors = {
                    "Happy": "#ffb6c1",
                    "Neutral": "#dda0dd",
                    "Sad": "#f7c6e8",
                    "Tired": "#d9b3ff",
                    "Anxious": "#b39ddb",
                    "Irritable": "#ffcdd2",
                    "Calm": "#d1c4e9",
                }

                # Add color legend
                legend_html = """
                <div style='margin-top:10px; display:flex; flex-wrap:wrap; gap:15px;'>
                """ + "".join(
                    [
                        f"<div style='display:flex; align-items:center; gap:5px; font-size:13px; color:#444;'>"
                        f"<span style='width:15px; height:15px; border-radius:50%; background:{mood_colors.get(mood, '#ffb6c1')}; display:inline-block;'></span>"
                        f"{mood}</div>"
                        for mood in mood_counts.index
                    ]
                ) + "</div>"

                st.markdown(legend_html, unsafe_allow_html=True)

                st.caption("💡 Each color represents a recorded mood in your cycle history 🌸")

            # Cycle Timeline
            st.markdown('<div class="glass-card"><h4>Cycle Timeline</h4></div>', unsafe_allow_html=True)
            if "date_start" in sorted_logs.columns and "date_end" in sorted_logs.columns:
                sorted_logs["date_start"] = pd.to_datetime(sorted_logs["date_start"], errors="coerce")
                sorted_logs["date_end"] = pd.to_datetime(sorted_logs["date_end"], errors="coerce")

                valid_cycles = sorted_logs.dropna(subset=["date_start", "date_end"]).sort_values("date_start")

                if not valid_cycles.empty:
                    for i, row in enumerate(valid_cycles.itertuples(), start=1):
                        # Format dates as: "13 Sep – 18 Sep, 2025"
                        start_str = row.date_start.strftime("%d %b")  # 13 Sep
                        end_str = row.date_end.strftime("%d %b, %Y")  # 18 Sep, 2025

                        st.markdown(
                            f"""
                            <div style='
                                background-color:#fff;
                                border-radius:12px;
                                padding:12px 18px;
                                margin-bottom:10px;
                                box-shadow: 0 2px 6px rgba(255,192,203,0.3);
                                font-size:14px;
                                color:#444;
                                '>
                                <b>Cycle {i}:</b> 🌸 {start_str} – {end_str}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.caption("📅 Each entry represents a complete recorded menstrual cycle 🌸")
                else:
                    st.info("No valid cycle data found. Log at least one full cycle to view timeline 🌷")
            else:
                st.warning("Date columns not found. Please verify your dataset structure.")

            # Friendly Insights
            insights = []
            if avg_cycle > 0:
                if variation == "Low":
                    insights.append("Your cycles are quite regular — good hormonal rhythm 🌸")
                else:
                    insights.append("Your cycles vary — track patterns for insights 💕")
            if most_common_display != "None":
                insights.append(f"{most_common} is your most common symptom — try hydration + warm compress 🫶")
            if insights:
                insight_text = "<br>".join(insights)
                st.markdown(f'<div class="glass-card"><p>{insight_text}</p></div>', unsafe_allow_html=True)

    # --- Export Data as CSV ---
    st.markdown("---")
    st.subheader("📂 Export Your Data")

    if not logs.empty:
        # Clean the dataset
        cleaned_df = logs.dropna(axis=1, how='all')  # Drop completely empty columns
        cleaned_df.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_df.columns]  # Normalize headers

        # Separate period logs and water logs for better organization
        period_columns = ['user_id', 'date_start', 'date_end', 'flow', 'mood', 'symptoms', 'notes']
        available_period_columns = [col for col in period_columns if col in cleaned_df.columns]
        period_logs = cleaned_df.dropna(subset=['date_start', 'date_end'])[available_period_columns]
        water_logs_all = cleaned_df[cleaned_df['date_start'].isna() & cleaned_df['date'].notna()][['date', 'water_glasses']]

        # Aggregate water logs by date to show daily totals
        if not water_logs_all.empty:
            water_daily_totals = water_logs_all.groupby(water_logs_all['date'].dt.date)['water_glasses'].sum().reset_index()
            water_daily_totals.columns = ['date', 'total_water_glasses']
            water_daily_totals['date'] = pd.to_datetime(water_daily_totals['date'])
        else:
            water_daily_totals = pd.DataFrame(columns=['date', 'total_water_glasses'])

        # Use the same average cycle length calculation as the dashboard insights
        avg_cycle_length = round(avg_cycle, 1) if cycle_lengths else "N/A"

        most_common_mood = (
            period_logs['mood'].mode()[0] if 'mood' in period_logs.columns and not period_logs['mood'].isna().all() else "N/A"
        )

        most_common_symptom = (
            period_logs['symptoms'].mode()[0] if 'symptoms' in period_logs.columns and not period_logs['symptoms'].isna().all() else "N/A"
        )

        total_logged_cycles = len(period_logs) if not period_logs.empty else 0
        total_water_days = len(water_daily_totals) if not water_daily_totals.empty else 0
        avg_daily_water = round(water_daily_totals['total_water_glasses'].mean(), 1) if not water_daily_totals.empty else "N/A"

        # Create summary text block
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

        # Convert DataFrames to CSV (with summary)
        buffer = io.StringIO()

        # Write period logs section
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
        csv_bytes = buffer.getvalue().encode('utf-8')

        # Dynamic export name
        export_name = f"ritucare_cycle_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

        # Download button
        st.download_button(
            label="📥 Download Cleaned CSV File",
            data=csv_bytes,
            file_name=export_name,
            mime="text/csv",
            use_container_width=True,
            help="Click to download your cleaned and summarized RituCare data 🌸"
        )

        st.caption("✨ Your file now includes aggregated daily water totals, detailed entries, and summary statistics!")
    else:
        st.warning("⚠️ No data available to export. Please log your cycles first.")

    # --- Data Management: Clear Logged Data ---
    st.markdown("---")
    st.markdown("### 🧹 Clear Logged Data")

    st.write(
        "You can clear all manually logged data (from your Log Period or Water Tracker entries). "
        "This will not affect your base datasets or analytics insights."
    )

    # Path to user log file
    log_file_path = "logs/user_cycle_log.csv"

    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as file:
            content = file.readlines()
            entry_count = max(0, len(content) - 1) if len(content) > 1 else 0

        st.info(f"📄 You currently have **{entry_count}** logged entries stored.")
    else:
        st.warning("⚠️ No log file found. You haven’t logged any data yet.")

    # Confirmation before clearing
    confirm_clear = st.checkbox("I confirm I want to clear all logged data")
    if st.button("🧼 Clear All Logged Data"):
        if confirm_clear:
            if os.path.exists(log_file_path):
                # Keep only the headers
                try:
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        header = f.readline()
                    with open(log_file_path, "w", encoding="utf-8") as f:
                        f.write(header)
                    st.success("✅ All manually logged data has been cleared successfully!")
                    st.rerun()  # Refresh to update the entry count
                except Exception as e:
                    st.error(f"❌ Failed to clear log data: {e}")
            else:
                st.warning("⚠️ No log file found to clear.")
        else:
            st.warning("Please confirm by checking the box above before clearing.")

st.markdown('<div class="footer">RituCare 🌸 Wellness Assistant — Student Research Project | Not Medical Advice</div>', unsafe_allow_html=True)
