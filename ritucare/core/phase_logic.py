from datetime import datetime, timedelta

def get_cycle_day(last_period_date, cycle_length=28):
    today = datetime.today().date()
    if not last_period_date:
        return None
    days_passed = (today - last_period_date).days
    return (days_passed % cycle_length) + 1

def get_cycle_phase(cycle_day):
    if cycle_day is None:
        return "Unknown", "❓"
    if cycle_day <= 5:
        return "Menstrual", "🩸"
    elif 6 <= cycle_day <= 13:
        return "Follicular", "🌱"
    elif 14 <= cycle_day <= 16:
        return "Ovulation", "✨"
    else:
        return "Luteal", "🌙"

def get_phase(cycle_day):
    phase, emoji = get_cycle_phase(cycle_day)
    return f"{phase} {emoji}"
