import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
import json
import gspread
from google.oauth2.service_account import Credentials
import os

# =========================
# UNIT CONVERSION HELPERS
# =========================

def lbs_to_kg(lbs: float) -> float:
    try:
        return round(lbs * 0.45359237, 2)
    except Exception:
        return 0.0

def kg_to_lbs(kg: float) -> float:
    try:
        return round(kg / 0.45359237, 2)
    except Exception:
        return 0.0

def in_to_cm(i: float) -> float:
    try:
        return round(i * 2.54, 2)
    except Exception:
        return 0.0

def cm_to_in(cm: float) -> float:
    try:
        return round(cm / 2.54, 2)
    except Exception:
        return 0.0


# ======================================================
# BASIC CONFIG – PROGRAM DEFINITION
# ======================================================

AI_ENABLED = True  # flip to False if you ever want to hard-disable AI

# 4-day template with equipment + muscles
PROGRAM = {
    "Upper 1": [
        {
            "name": "Flat Bench – Smith",
            "equipment": "Smith machine",
            "muscles": "Chest, triceps, front delts",
            "compound": True,
        },
        {
            "name": "Lat Pulldown – Wide Grip",
            "equipment": "Lat pulldown machine",
            "muscles": "Lats, upper back, biceps",
            "compound": True,
        },
        {
            "name": "Seated Row – Cable",
            "equipment": "Cable row machine",
            "muscles": "Mid-back, lats, rear delts, biceps",
            "compound": True,
        },
        {
            "name": "Lateral Raise – Dumbbells",
            "equipment": "Dumbbells",
            "muscles": "Side delts",
            "compound": False,
        },
    ],
    "Lower 1": [
        {
            "name": "Leg Press",
            "equipment": "Leg press machine",
            "muscles": "Quads, glutes, hamstrings",
            "compound": True,
        },
        {
            "name": "Romanian Deadlift – Smith",
            "equipment": "Smith machine",
            "muscles": "Hamstrings, glutes, lower back",
            "compound": True,
        },
        {
            "name": "Leg Curl",
            "equipment": "Leg curl machine",
            "muscles": "Hamstrings",
            "compound": False,
        },
        {
            "name": "Calf Raise – Machine",
            "equipment": "Calf raise machine",
            "muscles": "Calves",
            "compound": False,
        },
    ],
    "Upper 2": [
        {
            "name": "Incline Bench – Smith",
            "equipment": "Smith machine",
            "muscles": "Upper chest, shoulders, triceps",
            "compound": True,
        },
        {
            "name": "Pull-Down – Neutral Grip",
            "equipment": "Lat pulldown machine",
            "muscles": "Lats, biceps",
            "compound": True,
        },
        {
            "name": "Row – Chest Supported Machine",
            "equipment": "Row machine",
            "muscles": "Back, rear delts, biceps",
            "compound": True,
        },
        {
            "name": "Bicep Curl – Cable",
            "equipment": "Cable",
            "muscles": "Biceps",
            "compound": False,
        },
    ],
    "Lower 2": [
        {
            "name": "Hack Squat / Squat Machine",
            "equipment": "Hack squat machine",
            "muscles": "Quads, glutes",
            "compound": True,
        },
        {
            "name": "Hip Thrust – Machine",
            "equipment": "Hip thrust machine",
            "muscles": "Glutes, hamstrings",
            "compound": True,
        },
        {
            "name": "Leg Extension",
            "equipment": "Leg extension machine",
            "muscles": "Quads",
            "compound": False,
        },
        {
            "name": "Calf Raise – Seated",
            "equipment": "Calf raise machine",
            "muscles": "Calves",
            "compound": False,
        },
    ],
}


# ======================================================
# GOOGLE SHEETS LOGGING HELPERS
# ======================================================

def get_log_worksheet():
    """
    Connect to the Google Sheet and return the 'log' worksheet.
    Uses:
    - st.secrets['GOOGLE_SERVICE_ACCOUNT_JSON']
    - st.secrets['GOOGLE_SHEET_ID']
    """
    sa_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
    ws = sheet.worksheet("log")
    return ws


def _empty_log_df():
    return pd.DataFrame(
        columns=[
            "datetime",
            "day",
            "exercise",
            "equipment",
            "plate_weight_lbs",
            "base_weight_lbs",
            "base_type",
            "total_weight_lbs",
            "reps",
            "difficulty",
            "goal",
            "user_bodyweight_kg",
        ]
    )


def load_log():
    """
    Load the full workout log from the Google Sheet.
    """
    try:
        ws = get_log_worksheet()
        data = ws.get_all_values()
        if not data:
            return _empty_log_df()

        headers = data[0]
        rows = data[1:]
        if not rows:
            return _empty_log_df()

        df = pd.DataFrame(rows, columns=headers)

        # Ensure equipment column exists
        if "equipment" not in df.columns:
            df["equipment"] = ""

        # Convert types
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for col in [
            "plate_weight_lbs",
            "base_weight_lbs",
            "total_weight_lbs",
            "reps",
            "difficulty",
            "user_bodyweight_kg",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        st.warning(f"Error loading log from Google Sheets: {e}")
        return _empty_log_df()


def save_log_row(row_dict):
    """
    Append a single row (dict) to the Google Sheet 'log' worksheet.
    """
    try:
        ws = get_log_worksheet()

        existing = ws.get_all_values()
        expected_cols = [
            "datetime",
            "day",
            "exercise",
            "equipment",
            "plate_weight_lbs",
            "base_weight_lbs",
            "base_type",
            "total_weight_lbs",
            "reps",
            "difficulty",
            "goal",
            "user_bodyweight_kg",
        ]

        if not existing:
            ws.append_row(expected_cols)
        else:
            header = existing[0]
            if len(header) != len(expected_cols) or header != expected_cols:
                # overwrite header row if mismatched
                ws.update("A1", [expected_cols])

        dt_val = row_dict.get("datetime")
        if isinstance(dt_val, datetime):
            dt_str = dt_val.isoformat()
        else:
            dt_str = str(dt_val) if dt_val is not None else ""

        row_values = [
            dt_str,
            row_dict.get("day", ""),
            row_dict.get("exercise", ""),
            row_dict.get("equipment", ""),
            row_dict.get("plate_weight_lbs", ""),
            row_dict.get("base_weight_lbs", ""),
            row_dict.get("base_type", ""),
            row_dict.get("total_weight_lbs", ""),
            row_dict.get("reps", ""),
            row_dict.get("difficulty", ""),
            row_dict.get("goal", ""),
            row_dict.get("user_bodyweight_kg", ""),
        ]

        ws.append_row(row_values)

    except Exception as e:
        st.warning(f"Error saving log row to Google Sheets: {e}")


def get_last_entries(log_df, day, exercise_name):
    if log_df.empty:
        return None
    sub = log_df[(log_df["day"] == day) & (log_df["exercise"] == exercise_name)]
    if sub.empty:
        return None
    return sub.sort_values("datetime")


# ======================================================
# TRAINING LOGIC HELPERS
# ======================================================

def recommended_sets_reps_rest(goal, base_sets, base_reps, compound=True):
    if goal == "Strength":
        sets = base_sets + 1
        reps = max(3, min(base_reps, 6))
        rest = 180 if compound else 120
    elif goal == "Muscle Growth":
        sets = base_sets + 1
        reps = max(8, min(base_reps + 2, 15))
        rest = 90 if compound else 60
    elif goal == "Endurance":
        sets = base_sets
        reps = max(12, base_reps + 4)
        rest = 45
    else:  # General Fitness
        sets = base_sets
        reps = base_reps
        rest = 60 if compound else 45
    return sets, reps, rest


def estimate_1rm_from_log(sub_df):
    if sub_df is None or sub_df.empty:
        return None
    best_est = None
    for _, row in sub_df.iterrows():
        w = row.get("total_weight_lbs", 0)
        r = row.get("reps", 0)
        if w <= 0 or r <= 0:
            continue
        est = w * (1 + r / 30.0)
        if best_est is None or est > best_est:
            best_est = est
    return best_est


def suggest_weight_from_1rm(est_1rm_lbs, goal):
    if est_1rm_lbs is None:
        return None
    if goal == "Strength":
        pct = 0.82
    elif goal == "Muscle Growth":
        pct = 0.72
    elif goal == "Endurance":
        pct = 0.60
    else:
        pct = 0.70
    return est_1rm_lbs * pct


def suggest_weight_from_last(last_df, goal):
    if last_df is None or last_df.empty:
        return None
    last = last_df.iloc[-1]
    last_total = last.get("total_weight_lbs", None)
    if last_total is None:
        return None
    if goal == "Strength":
        return last_total + 5
    elif goal == "Muscle Growth":
        return last_total + 2.5
    else:
        return last_total


def hybrid_weight_suggestion(log_df, day, exercise_name, goal):
    sub = get_last_entries(log_df, day, exercise_name)
    est_1rm = estimate_1rm_from_log(sub)
    if est_1rm is not None:
        return suggest_weight_from_1rm(est_1rm, goal), est_1rm
    else:
        simple = suggest_weight_from_last(sub, goal)
        return simple, None


# ======================================================
# TRAINING SUMMARY FOR AI (LOG-AWARE)
# ======================================================

def build_training_summary_for_day(day: str) -> str:
    df = load_log()
    if df.empty:
        return "No training data logged yet."

    if day not in PROGRAM:
        return f"No program definition found for day '{day}'."

    df_day = df[df["day"] == day].sort_values("datetime")
    if df_day.empty:
        return f"No logged sets yet for {day}."

    lines = [f"Recent training summary for {day} (up to last 3 sets per exercise):"]
    for ex in PROGRAM[day]:
        ex_name = ex["name"]
        sub = df_day[df_day["exercise"] == ex_name].sort_values("datetime").tail(3)
        if sub.empty:
            continue

        last = sub.iloc[-1]
        last_dt = last["datetime"]
        if isinstance(last_dt, str):
            try:
                last_dt = pd.to_datetime(last_dt)
            except Exception:
                pass

        try:
            last_date_str = last_dt.strftime("%Y-%m-%d %H:%M") if not pd.isna(last_dt) else "unknown time"
        except Exception:
            last_date_str = "unknown time"

        best_weight = sub["total_weight_lbs"].max()
        best_reps = sub["reps"].max()
        avg_diff = sub["difficulty"].mean() if "difficulty" in sub.columns else None

        line = (
            f"- {ex_name}: last set {last['total_weight_lbs']} lbs x {last['reps']} reps"
            f" at difficulty {last.get('difficulty', 'N/A')}/10 on {last_date_str}; "
            f"best in last {len(sub)} sets was {best_weight} lbs x {best_reps} reps"
        )
        if avg_diff is not None:
            line += f", average difficulty ~{avg_diff:.1f}/10."
        else:
            line += "."

        lines.append(line)

    if len(lines) == 1:
        return f"No detailed exercise entries found for {day}, but the day exists in the program."

    return "\n".join(lines)


# ======================================================
# AI CHAT (OPENAI)
# ======================================================

def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def ai_trainer_reply(user_message: str, day: str, goal: str, profile: dict) -> str:
    if not AI_ENABLED:
        return "AI trainer is not enabled yet."

    log_summary = build_training_summary_for_day(day)

    try:
        client = OpenAI()  # uses OPENAI_API_KEY from environment / secrets

        system_prompt = f"""
You are a clear, direct lifting coach.

User profile:
- Age: {profile.get('age')}
- Sex: {profile.get('sex')}
- Experience: {profile.get('experience')}
- Height (cm): {profile.get('height_cm')}
- Bodyweight (kg): {profile.get('bodyweight_kg')}
- Preferred duration: {profile.get('preferred_duration')}
- Split: {profile.get('preferred_split')}
- Injuries/Limitations: {profile.get('injuries')}
- Notes: {profile.get('notes')}

Current workout day: {day}
Current goal: {goal}

Recent training data:
{log_summary}

Coaching rules:
- Talk like you would to a lifter in the gym. Clear and direct.
- Give specific sets, reps, rest ranges, and weight strategy.
- If no log data, give solid starting points.
- If log exists, use it: mention progress, suggest next steps.
- Be concise, not essay-length.
""".strip()

        messages = [{"role": "system", "content": system_prompt}]

        for msg in st.session_state["chat_history"]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error talking to AI trainer: {e}"


def handle_send_chat(day: str, goal: str, profile: dict):
    msg = st.session_state.get("chat_input", "")
    msg = msg.strip()
    if not msg:
        return
    st.session_state["chat_history"].append({"role": "user", "content": msg})
    reply = ai_trainer_reply(msg, day, goal, profile)
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


# ======================================================
# MAIN STREAMLIT APP
# ======================================================

def main():
    st.set_page_config(page_title="System Stress Test", page_icon="💪", layout="wide")
    init_chat_state()

    # ----- SIDEBAR: USER PROFILE & SETTINGS -----
    st.sidebar.title("User Profile & Settings")

    unit_system = st.sidebar.selectbox("Unit System", ["Imperial (lbs/in)", "Metric (kg/cm)"], index=0)
    st.session_state["unit_system"] = unit_system

    if "raw_height_cm" not in st.session_state:
        st.session_state["raw_height_cm"] = 180.0
    if "raw_weight_kg" not in st.session_state:
        st.session_state["raw_weight_kg"] = 80.0

    if unit_system.startswith("Imperial"):
        show_height = cm_to_in(st.session_state["raw_height_cm"])
        show_weight = kg_to_lbs(st.session_state["raw_weight_kg"])
        height_label = "Height (inches)"
        weight_label = "Body weight (lbs)"
        height_min, height_max = 40.0, 90.0
        weight_min, weight_max = 80.0, 600.0
    else:
        show_height = st.session_state["raw_height_cm"]
        show_weight = st.session_state["raw_weight_kg"]
        height_label = "Height (cm)"
        weight_label = "Body weight (kg)"
        height_min, height_max = 130.0, 230.0
        weight_min, weight_max = 40.0, 300.0

    with st.sidebar.expander("Profile", expanded=True):
        age = st.number_input("Age", min_value=10, max_value=80, value=19, step=1)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=0)
        experience = st.selectbox("Experience Level", ["Novice", "Intermediate", "Advanced"], index=0)

        new_height = st.number_input(
            height_label,
            min_value=height_min,
            max_value=height_max,
            value=float(show_height),
            step=0.5,
        )
        new_weight = st.number_input(
            weight_label,
            min_value=weight_min,
            max_value=weight_max,
            value=float(show_weight),
            step=0.5,
        )

        preferred_duration = st.selectbox(
            "Preferred Session Duration",
            [
                "30 min (quick)",
                "1 hour",
                "2 hours",
                "3 hours",
                "4 hours",
                "5 hours",
                "Flexible",
            ],
            index=2,
        )

        preferred_split = st.selectbox(
            "Preferred Split",
            ["Upper/Lower", "Push/Pull/Legs", "Full Body", "Custom"],
            index=0,
        )

        injuries = st.text_area("Injuries / limitations (optional)", height=60)
        notes = st.text_area("Notes / preferences (optional)", height=60)

    if unit_system.startswith("Imperial"):
        st.session_state["raw_height_cm"] = in_to_cm(new_height)
        st.session_state["raw_weight_kg"] = lbs_to_kg(new_weight)
    else:
        st.session_state["raw_height_cm"] = new_height
        st.session_state["raw_weight_kg"] = new_weight

    height_cm = st.session_state["raw_height_cm"]
    bodyweight_kg = st.session_state["raw_weight_kg"]

    if height_cm < 130 or height_cm > 230:
        st.sidebar.warning("Height looks unusual. Double-check the value and unit system.")
    if bodyweight_kg < 40 or bodyweight_kg > 250:
        st.sidebar.warning("Body weight looks unusual. Double-check the value and unit system.")

    bmi = None
    if height_cm > 0:
        bmi = bodyweight_kg / ((height_cm / 100.0) ** 2)

    profile = {
        "age": age,
        "sex": sex,
        "experience": experience,
        "height_cm": height_cm,
        "bodyweight_kg": bodyweight_kg,
        "preferred_duration": preferred_duration,
        "preferred_split": preferred_split,
        "injuries": injuries,
        "notes": notes,
        "unit_system": unit_system,
        "bmi": bmi,
    }
    st.session_state["user_profile"] = profile

    if bmi is not None:
        st.sidebar.metric("BMI (approx)", f"{bmi:.1f}")

    goal = st.sidebar.selectbox(
        "Training Goal",
        ["Strength", "Muscle Growth", "Endurance", "General Fitness"],
        index=1,
    )
    day_options = list(PROGRAM.keys())
    today_day = st.sidebar.selectbox("Today is:", day_options, index=0)
    show_log_table = st.sidebar.checkbox("Show log table at bottom", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Trainer Chat (AI)")

    chat_box = st.sidebar.container(height=250, border=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            chat_box.markdown(f"**You:** {msg['content']}")
        else:
            chat_box.markdown(f"**Coach:** {msg['content']}")

    st.sidebar.text_area("Message to coach", key="chat_input", height=80)
    st.sidebar.button(
        "Send to coach",
        on_click=handle_send_chat,
        args=(today_day, goal, profile),
    )

    # ----- MAIN HEADER -----
    now = datetime.now()
    st.title(f"System Stress Test: {today_day}")
    st.caption(
        f"Date: {now.strftime('%Y-%m-%d')} | Time: {now.strftime('%H:%M:%S')} | "
        f"Goal: {goal} | Units: {unit_system}"
    )

    tab_workout, tab_history = st.tabs(["Workout", "History / Charts"])

    log_df = load_log()

    # ======================================================
    # TAB 1 – WORKOUT
    # ======================================================
    with tab_workout:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Today's Task List")
            st.write("Suggested plan + logging. Each click saves one set to your log.")

            for ex in PROGRAM[today_day]:
                ex_name = ex["name"]
                equipment = ex.get("equipment", "")
                muscles = ex.get("muscles", "")
                compound = ex.get("compound", True)

                st.markdown(f"### {ex_name}")
                st.write(f"**Equipment:** {equipment}")
                st.write(f"**Muscles:** {muscles}")

                with st.expander("Form & posture (basic)", expanded=False):
                    st.write(
                        "Brace your core, control the weight, no bouncing. "
                        "Use a full range of motion you can control without joint pain."
                    )

                base_sets = 3
                base_reps = 8
                rec_sets, rec_reps, rec_rest = recommended_sets_reps_rest(
                    goal, base_sets, base_reps, compound
                )

                st.write("**Suggested plan:**")
                st.write(f"- Sets per exercise: {rec_sets}")
                st.write(f"- Reps per set: {rec_reps}")
                st.write(f"- Rest between sets: {rec_rest} seconds")

                last_df = get_last_entries(log_df, today_day, ex_name)
                if last_df is not None and not last_df.empty:
                    last = last_df.iloc[-1]
                    dt_str = last["datetime"]
                    if isinstance(dt_str, datetime):
                        when_str = dt_str.strftime("%Y-%m-%d %H:%M")
                    else:
                        try:
                            when_str = pd.to_datetime(dt_str).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            when_str = "unknown"

                    st.info(
                        f"Last logged: {last.get('total_weight_lbs', 0)} lbs "
                        f"x {last.get('reps', 0)} reps @ difficulty "
                        f"{last.get('difficulty', 'N/A')}/10 on {when_str}."
                    )

                suggested_total_lbs, est_1rm_lbs = hybrid_weight_suggestion(
                    log_df, today_day, ex_name, goal
                )
                if est_1rm_lbs is not None:
                    st.write(f"Estimated 1RM from history: ~{est_1rm_lbs:.1f} lbs")
                if suggested_total_lbs is not None:
                    st.write(
                        f"Suggested working total weight today: ~{suggested_total_lbs:.1f} lbs"
                    )

                st.write("**Log a set for this exercise**")

                plate_weight = st.number_input(
                    f"Plate weight added (plates or pin) – {ex_name} (lbs)",
                    min_value=0.0,
                    max_value=2000.0,
                    step=2.5,
                    key=f"plate_{today_day}_{ex_name}",
                )

                base_type = st.selectbox(
                    f"Base weight type for {ex_name}",
                    ["Starting weight", "Starting resistance"],
                    key=f"base_type_{today_day}_{ex_name}",
                    help="Label taken from the machine that describes its built-in starting load.",
                )

                default_base = 0.0
                if last_df is not None and not last_df.empty:
                    default_base = float(last_df.iloc[-1].get("base_weight_lbs", 0.0))

                base_weight = st.number_input(
                    f"Machine base ({base_type}) for {ex_name} (lbs, 0 if none)",
                    min_value=0.0,
                    max_value=500.0,
                    step=1.0,
                    value=default_base,
                    key=f"base_{today_day}_{ex_name}",
                )

                total_weight = plate_weight + base_weight
                st.write(f"Total load this set (added + base): **{total_weight:.1f} lbs**")

                reps = st.number_input(
                    f"Reps completed for {ex_name}",
                    min_value=0,
                    max_value=50,
                    step=1,
                    key=f"reps_{today_day}_{ex_name}",
                )

                difficulty = st.number_input(
                    f"Difficulty (1–10, can be decimal) for {ex_name}",
                    min_value=1.0,
                    max_value=10.0,
                    step=0.5,
                    key=f"diff_{today_day}_{ex_name}",
                )

                if st.button(
                    f"Save set for {ex_name}", key=f"save_{today_day}_{ex_name}"
                ):
                    if reps <= 0 or total_weight <= 0:
                        st.warning("Enter a positive total weight and reps before saving.")
                    else:
                        row = {
                            "datetime": datetime.now(),
                            "day": today_day,
                            "exercise": ex_name,
                            "equipment": equipment,
                            "plate_weight_lbs": plate_weight,
                            "base_weight_lbs": base_weight,
                            "base_type": base_type,
                            "total_weight_lbs": total_weight,
                            "reps": reps,
                            "difficulty": difficulty,
                            "goal": goal,
                            "user_bodyweight_kg": bodyweight_kg,
                        }
                        save_log_row(row)
                        st.success("Set saved to log.")

        with col_right:
            st.subheader("Muscle Map & Legend")

            img_path = "muscles_front_back.png"
            if os.path.exists(img_path):
                st.image(
                    img_path,
                    caption="Front & Back muscle groups",
                    use_container_width=True,
                )
            else:
                st.info(
                    "Place 'muscles_front_back.png' in this folder to show the muscle diagram here."
                )

            with st.expander("Legend: Terms & Scales", expanded=False):
                st.markdown(
                    """
**Sets** – How many times you repeat the exercise block.  
**Reps** – Number of times you do the movement in one set.  

**Difficulty (1–10)**  
1–3 = Easy / warm-up  
4–6 = Working, could do more reps  
7–9 = Hard, 1–2 reps left  
10 = No extra reps left  

**Weight fields (always in lbs)**  
- **Plate weight** – Plates or pin setting you add.  
- **Base weight** – Machine's labeled starting weight or resistance.  
- **Total load** – Added + base.  

**Goals**  
- **Strength** = Heavier weight, lower reps, longer rest.  
- **Muscle Growth** = Moderate weight, 8–15 reps, moderate rest.  
- **Endurance** = Lighter weight, higher reps, shorter rest.  
- **General Fitness** = Balanced approach.  
"""
                )

    # ======================================================
    # TAB 2 – HISTORY / CHARTS
    # ======================================================
    with tab_history:
        st.subheader("Bodyweight Trend")

        if log_df.empty or "user_bodyweight_kg" not in log_df.columns:
            st.write("No bodyweight data logged yet.")
        else:
            bw_df = log_df.dropna(subset=["user_bodyweight_kg"]).copy()
            if not bw_df.empty:
                bw_df = bw_df.sort_values("datetime")
                display_col = "user_bodyweight_kg"
                label = "Bodyweight (kg)"
                if unit_system.startswith("Imperial"):
                    bw_df["bodyweight_lbs"] = bw_df["user_bodyweight_kg"].apply(kg_to_lbs)
                    display_col = "bodyweight_lbs"
                    label = "Bodyweight (lbs)"
                bw_df = bw_df.set_index("datetime")
                st.line_chart(bw_df[display_col], use_container_width=True)
                st.caption(label)
            else:
                st.write("No bodyweight data logged yet.")

        st.markdown("---")
        st.subheader("Per-exercise Progress")

        if log_df.empty:
            st.write("No sets logged yet.")
        else:
            ex_names = sorted(log_df["exercise"].dropna().unique())
            ex_sel = st.selectbox("Select exercise for charts", ex_names)

            sub = log_df[log_df["exercise"] == ex_sel].sort_values("datetime")
            if sub.empty:
                st.write("No data for this exercise yet.")
            else:
                sub = sub.copy()
                sub["volume"] = sub["total_weight_lbs"] * sub["reps"]

                est_list = []
                for _, r in sub.iterrows():
                    w = r["total_weight_lbs"]
                    rep = r["reps"]
                    if w > 0 and rep > 0:
                        est_list.append(w * (1 + rep / 30.0))
                    else:
                        est_list.append(None)
                sub["est_1rm_lbs"] = est_list

                st.markdown("#### Progress for " + ex_sel)

                c1, c2 = st.columns(2)

                with c1:
                    st.write("Weight over time (total load, lbs)")
                    chart_df = sub[["datetime", "total_weight_lbs"]].set_index("datetime")
                    st.line_chart(chart_df, use_container_width=True)

                    st.write("Volume (total load × reps, lbs×reps)")
                    chart_vol = sub[["datetime", "volume"]].set_index("datetime")
                    st.line_chart(chart_vol, use_container_width=True)

                with c2:
                    st.write("Difficulty trend")
                    diff_df = sub[["datetime", "difficulty"]].set_index("datetime")
                    st.line_chart(diff_df, use_container_width=True)

                    st.write("Estimated 1RM over time (lbs)")
                    est_df = sub[["datetime", "est_1rm_lbs"]].set_index("datetime")
                    st.line_chart(est_df, use_container_width=True)

        if show_log_table:
            st.markdown("---")
            st.subheader("Full log (last 200 rows)")
            if log_df.empty:
                st.write("No sets logged yet.")
            else:
                st.dataframe(log_df.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
