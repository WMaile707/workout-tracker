import streamlit as st
import pandas as pd
from datetime import datetime
import os
from openai import OpenAI  # OpenAI client

import json
import gspread
from google.oauth2.service_account import Credentials

# ======================================================
# CONFIG
# ======================================================
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


def load_log():
    """
    Load the full workout log from the Google Sheet.

    If the sheet is empty, return an empty DataFrame
    with the expected columns.
    """
    try:
        ws = get_log_worksheet()
        data = ws.get_all_values()
        if not data:
            # No rows yet
            return pd.DataFrame(
                columns=[
                    "datetime",
                    "day",
                    "exercise",
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

        # First row is header
        headers = data[0]
        rows = data[1:]
        if not rows:
            return pd.DataFrame(columns=headers)

        df = pd.DataFrame(rows, columns=headers)

        # Convert types
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for col in ["plate_weight_lbs", "base_weight_lbs", "total_weight_lbs", "reps", "difficulty", "user_bodyweight_kg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        # On error, return empty DF so the app still runs
        st.warning(f"Error loading log from Google Sheets: {e}")
        return pd.DataFrame(
            columns=[
                "datetime",
                "day",
                "exercise",
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


def save_log_row(row_dict):
    """
    Append a single row (dict) to the Google Sheet 'log' worksheet.
    If sheet has no header yet, write header first.
    """
    try:
        ws = get_log_worksheet()

        # Get existing data to see if header exists
        existing = ws.get_all_values()
        expected_cols = [
            "datetime",
            "day",
            "exercise",
            "plate_weight_lbs",
            "base_weight_lbs",
            "base_type",
            "total_weight_lbs",
            "reps",
            "difficulty",
            "goal",
            "user_bodyweight_kg",
        ]

        # If sheet empty, write header row
        if not existing:
            ws.append_row(expected_cols)
        else:
            # If header exists but is different/short, you could adjust here if needed
            pass

        # Ensure datetime is string
        dt_val = row_dict.get("datetime")
        if isinstance(dt_val, datetime):
            dt_str = dt_val.isoformat()
        else:
            dt_str = str(dt_val) if dt_val is not None else ""

        # Build row in correct column order
        row_values = [
            dt_str,
            row_dict.get("day", ""),
            row_dict.get("exercise", ""),
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
    """
    Build a short text summary of recent training for the current day
    (Upper 1, Lower 1, etc.) to feed into the AI.
    """
    df = load_log()
    if df.empty:
        return "No training data logged yet in workout_log.csv."

    if day not in PROGRAM:
        return f"No program definition found for day '{day}'."

    # Filter by this day
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
# AI CHAT (REAL OPENAI INTEGRATION)
# ======================================================

def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def ai_trainer_reply(user_message: str, day: str, goal: str, profile: dict):
    """
    Real AI trainer using OpenAI Chat Completions.
    Uses:
    - st.session_state["chat_history"] for memory within the current app session.
    - build_training_summary_for_day(day) to be aware of your logged history.
    """
    if not AI_ENABLED:
        return (
            "AI trainer is not enabled yet.\n\n"
            "To enable it, set AI_ENABLED = True in the code and configure the OpenAI API."
        )

    # Build log-aware summary for this day
    log_summary = build_training_summary_for_day(day)

    try:
        client = OpenAI()  # uses OPENAI_API_KEY from your environment

        system_prompt = f"""
You are a friendly but direct lifting coach.

User profile:
- Age: {profile.get('age')}
- Sex: {profile.get('sex')}
- Experience: {profile.get('experience')}
- Height (cm): {profile.get('height_cm')}
- Bodyweight (kg): {profile.get('bodyweight_kg')}
- Preferred duration: {profile.get('preferred_duration')}
- Preferred split: {profile.get('preferred_split')}
- Injuries/limitations: {profile.get('injuries')}
- Notes: {profile.get('notes')}

Current workout day: {day}
Current goal: {goal}

Recent training data (from the app's workout_log.csv):
{log_summary}

Coaching rules:
- Talk like you would to a lifter in the gym: clear, direct, no fluff.
- Be concrete and specific: talk sets, reps, rest times, and weight strategy.
- If the log summary shows specific weights/reps, use them to guide progressions.
- If there is no log data yet, say that clearly and give general starting recommendations.
- If user mentions an exercise by name, give advice specific to that exercise.
- Suggest adjustments: more/less sets, reps, rest, and when to increase/decrease weight.
- Keep responses short enough to read quickly between sets (no essays).
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add past chat for memory (within this Streamlit session)
        for m in st.session_state["chat_history"]:
            role = "user" if m["role"] == "user" else "assistant"
            messages.append({"role": role, "content": m["content"]})

        # Add new user message
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
    """Callback to handle sending chat, keeping Streamlit happy."""
    msg = st.session_state.get("chat_input", "")
    if not msg.strip():
        return
    st.session_state["chat_history"].append({"role": "user", "content": msg.strip()})
    reply = ai_trainer_reply(msg.strip(), day, goal, profile)
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    st.session_state["chat_input"] = ""


# ======================================================
# MAIN STREAMLIT APP
# ======================================================

def main():
    st.set_page_config(page_title="System Stress Test", page_icon="💪", layout="wide")
    init_chat_state()

    # -------------------------
    # SIDEBAR – USER PROFILE & UNIT SYSTEM
    # -------------------------
    st.sidebar.title("User Profile & Settings")

    unit_system = st.sidebar.selectbox("Unit System", ["Imperial (lbs/in)", "Metric (kg/cm)"], index=0)
    st.session_state["unit_system"] = unit_system

    if "raw_height_cm" not in st.session_state:
        st.session_state.raw_height_cm = 179.0
    if "raw_weight_kg" not in st.session_state:
        st.session_state.raw_weight_kg = 105.0

    if unit_system.startswith("Imperial"):
        shown_height = cm_to_in(st.session_state.raw_height_cm)
        shown_weight = kg_to_lbs(st.session_state.raw_weight_kg)
        height_label = "Height (inches)"
        weight_label = "Body Weight (lbs)"
        height_min, height_max = 40.0, 100.0
        weight_min, weight_max = 70.0, 600.0
    else:
        shown_height = st.session_state.raw_height_cm
        shown_weight = st.session_state.raw_weight_kg
        height_label = "Height (cm)"
        weight_label = "Body Weight (kg)"
        height_min, height_max = 100.0, 250.0
        weight_min, weight_max = 30.0, 300.0

    with st.sidebar.expander("Profile", expanded=True):
        age = st.number_input("Age", min_value=10, max_value=100, value=19)

        new_height = st.number_input(
            height_label,
            min_value=height_min,
            max_value=height_max,
            value=float(shown_height),
            step=0.5,
            help="Switch units above; value auto-converts between inches and cm."
        )
        if unit_system.startswith("Imperial"):
            st.session_state.raw_height_cm = in_to_cm(new_height)
        else:
            st.session_state.raw_height_cm = new_height

        new_weight = st.number_input(
            weight_label,
            min_value=weight_min,
            max_value=weight_max,
            value=float(shown_weight),
            step=0.5,
            help="Switch units above; value auto-converts between lbs and kg."
        )
        if unit_system.startswith("Imperial"):
            st.session_state.raw_weight_kg = lbs_to_kg(new_weight)
        else:
            st.session_state.raw_weight_kg = new_weight

        sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=0)
        experience = st.selectbox("Experience Level", ["Novice", "Intermediate", "Advanced"], index=0)

        pref_duration = st.selectbox(
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

        pref_split = st.selectbox("Preferred Split", ["Upper/Lower", "Push/Pull/Legs", "Full Body", "Custom"], index=0)
        injuries = st.text_area("Injuries / limitations (optional)", height=60)
        notes = st.text_area("Notes / preferences (optional)", height=60)

        height_m = st.session_state.raw_height_cm / 100.0 if st.session_state.raw_height_cm > 0 else 0
        weight_kg = st.session_state.raw_weight_kg
        if height_m > 0 and weight_kg > 0:
            bmi = weight_kg / (height_m ** 2)
            st.metric("BMI (approx)", f"{bmi:.1f}")
        else:
            bmi = None

        if st.session_state.raw_height_cm < 130 or st.session_state.raw_height_cm > 220:
            st.warning("Height looks unusual. Double-check the value and unit system.")
        if st.session_state.raw_weight_kg < 40 or st.session_state.raw_weight_kg > 250:
            st.warning("Body weight looks unusual. Double-check the value and unit system.")

    profile = {
        "age": age,
        "height_cm": st.session_state.raw_height_cm,
        "bodyweight_kg": st.session_state.raw_weight_kg,
        "bmi": bmi,
        "sex": sex,
        "experience": experience,
        "preferred_duration": pref_duration,
        "preferred_split": pref_split,
        "injuries": injuries,
        "notes": notes,
        "unit_system": unit_system,
    }
    st.session_state["user_profile"] = profile

    goal = st.sidebar.selectbox("Training Goal", GOALS, index=1)
    day = st.sidebar.selectbox("Today is:", list(PROGRAM.keys()), index=0)
    show_log_table = st.sidebar.checkbox("Show log table at bottom", value=True)

    # -------------------------
    # SIDEBAR – TRAINER CHAT
    # -------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Trainer Chat (AI)")

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
        args=(day, goal, profile),
    )

    # -------------------------
    # HEADER
    # -------------------------
    now = datetime.now()
    st.title(f"System Stress Test: {day}")
    st.caption(
        f"Date: {now.strftime('%Y-%m-%d')} | Time: {now.strftime('%H:%M:%S')} | "
        f"Goal: {goal} | Units: {unit_system}"
    )

    tab_workout, tab_history = st.tabs(["Workout", "History / Charts"])

    log_df = load_log()

    # -------------------------
    # TAB 1 – WORKOUT
    # -------------------------
    with tab_workout:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Today's Task List")
            st.write("Suggested sets, reps, and rest based on your goal. You log what you actually do.")

            for ex in PROGRAM[day]:
                st.markdown("---")
                ex_name = ex["name"]
                st.markdown(f"### {ex_name}")
                st.write(f"**Equipment:** {ex['equipment']}")
                st.write(f"**Muscles:** {ex['muscles']}")
                with st.expander("Form & posture details", expanded=False):
                    st.write(ex["description"])

                rec_sets, rec_reps, rec_rest = recommended_sets_reps_rest(
                    goal, ex["base_sets"], ex["base_reps"], ex["compound"]
                )
                st.write("**Suggested plan:**")
                st.write(f"- Sets: {rec_sets}")
                st.write(f"- Reps per set: {rec_reps}")
                st.write(f"- Rest between sets: {rec_rest} seconds")

                last_df = get_last_entries(log_df, day, ex_name)
                if last_df is not None and not last_df.empty:
                    last = last_df.iloc[-1]
                    dt_str = last["datetime"].strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(last["datetime"]) else "unknown"
                    st.info(
                        f"Last logged: total {last['total_weight_lbs']} lbs "
                        f"(added {last['plate_weight_lbs']} + base {last['base_weight_lbs']}), "
                        f"{last['reps']} reps on {dt_str} "
                        f"(Difficulty {last['difficulty']}/10)"
                    )

                suggested_total_lbs, est_1rm_lbs = hybrid_weight_suggestion(log_df, day, ex_name, goal)
                if est_1rm_lbs is not None:
                    st.write(f"Estimated 1RM from history: ~{est_1rm_lbs:.1f} lbs")
                if suggested_total_lbs is not None:
                    st.write(f"Suggested working total weight today: ~{suggested_total_lbs:.1f} lbs")

                st.write("#### Log a set for this exercise")

                plate_weight = st.number_input(
                    f"Weight added (plates or pin setting) – {ex_name} (lbs)",
                    min_value=0.0,
                    step=5.0,
                    key=f"plate_{day}_{ex_name}",
                    help="Enter the weight you added (plates or pin). This is always in pounds."
                )

                base_type = st.selectbox(
                    f"Base weight type for {ex_name}",
                    ["Starting weight", "Starting resistance"],
                    key=f"basetype_{day}_{ex_name}",
                    help="Label from the machine that describes its built-in starting load."
                )

                default_base = 0.0
                if last_df is not None and not last_df.empty:
                    default_base = float(last_df.iloc[-1]["base_weight_lbs"])

                base_weight = st.number_input(
                    f"Machine base ({base_type}) for {ex_name} (lbs, 0 if none)",
                    min_value=0.0,
                    step=5.0,
                    value=default_base,
                    key=f"base_{day}_{ex_name}",
                )

                total_weight = plate_weight + base_weight
                st.write(f"**Total load this set (added + base): {total_weight:.1f} lbs**")

                reps = st.number_input(
                    f"Reps completed for {ex_name}",
                    min_value=0,
                    step=1,
                    key=f"reps_{day}_{ex_name}",
                )
                difficulty = st.number_input(
                    f"Difficulty (1–10, can be decimal) for {ex_name}",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.5,
                    key=f"diff_{day}_{ex_name}",
                )

                if st.button(f"Save set for {ex_name}", key=f"save_{day}_{ex_name}"):
                    if reps <= 0 or total_weight <= 0:
                        st.warning("Enter a positive total weight and reps before saving.")
                    else:
                        row = {
                            "datetime": datetime.now(),
                            "day": day,
                            "exercise": ex_name,
                            "plate_weight_lbs": plate_weight,
                            "base_weight_lbs": base_weight,
                            "base_type": base_type,
                            "total_weight_lbs": total_weight,
                            "reps": reps,
                            "difficulty": difficulty,
                            "goal": goal,
                            "user_bodyweight_kg": st.session_state.raw_weight_kg,
                        }
                        save_log_row(row)
                        st.success("Set saved to log.")

        with col_right:
            st.subheader("Muscle Map & Legend")

            if os.path.exists("muscles_front_back.png"):
                st.image(
                    "muscles_front_back.png",
                    caption="Front & Back muscle groups",
                    use_container_width=True
                )
            elif os.path.exists("muscles_front.png") or os.path.exists("muscles_back.png"):
                tabs = st.tabs(["Front", "Back"])
                if os.path.exists("muscles_front.png"):
                    with tabs[0]:
                        st.image("muscles_front.png", caption="Front muscle groups", use_container_width=True)
                if os.path.exists("muscles_back.png"):
                    with tabs[-1]:
                        st.image("muscles_back.png", caption="Back muscle groups", use_container_width=True)
            else:
                st.info(
                    "Place 'muscles_front_back.png' (or 'muscles_front.png' and 'muscles_back.png') "
                    "in this folder to show muscle diagrams here."
                )

            with st.expander("Legend: Terms & Scales", expanded=False):
                st.markdown(
                    """
                    **Sets** – How many times you repeat the exercise block.  
                    **Reps** – Number of times you do the movement in one set.  

                    **Difficulty (1–10):**  
                    - 1–4 = Easy / warm-up  
                    - 5–7 = Working, could do more reps  
                    - 8–9 = Hard, 1–2 reps left  
                    - 10 = Max effort, no reps left  

                    **Weight fields (always in lbs):**  
                    - **Weight added** – Plates or pin setting you add.  
                    - **Base weight** – Machine's labeled starting weight or resistance.  
                    - **Total load** – Added + base.  

                    **Goals:**  
                    - **Strength** – Heavier weight, lower reps, longer rest  
                    - **Muscle Growth** – Moderate weight, 8–15 reps, moderate rest  
                    - **Endurance** – Lighter weight, higher reps, shorter rest  
                    - **General Fitness** – Balanced approach  
                    """
                )

    # -------------------------
    # TAB 2 – HISTORY / CHARTS
    # -------------------------
    with tab_history:
        st.subheader("History & Progress Charts")

        if log_df.empty:
            st.write("No sets logged yet.")
        else:
            st.markdown("### Bodyweight Trend")
            bw_df = log_df.dropna(subset=["user_bodyweight_kg"]).copy()
            if not bw_df.empty:
                bw_df = bw_df.sort_values("datetime")
                bw_df["date"] = bw_df["datetime"].dt.date
                bw_daily = bw_df.groupby("date")["user_bodyweight_kg"].last().reset_index()

                if unit_system.startswith("Imperial"):
                    bw_daily["bodyweight_display"] = bw_daily["user_bodyweight_kg"].apply(kg_to_lbs)
                    bw_label = "Bodyweight (lbs)"
                else:
                    bw_daily["bodyweight_display"] = bw_daily["user_bodyweight_kg"]
                    bw_label = "Bodyweight (kg)"

                bw_daily = bw_daily.set_index("date")
                st.line_chart(bw_daily["bodyweight_display"], use_container_width=True)
                st.caption(bw_label)
            else:
                st.write("No bodyweight data logged yet.")

            st.markdown("---")
            st.markdown("### Exercise Progress")

            ex_names = sorted(log_df["exercise"].unique())
            ex_sel = st.selectbox("Select exercise for charts", ex_names)

            sub = log_df[log_df["exercise"] == ex_sel].sort_values("datetime")
            if sub.empty:
                st.write("No data for this exercise yet.")
            else:
                sub["date"] = sub["datetime"].dt.date
                sub["volume"] = sub["total_weight_lbs"] * sub["reps"]

                est_list = []
                for _, row in sub.iterrows():
                    w = row["total_weight_lbs"]
                    r = row["reps"]
                    if w > 0 and r > 0:
                        est_list.append(w * (1 + r / 30.0))
                    else:
                        est_list.append(None)
                sub["est_1rm_lbs"] = est_list

                st.markdown(f"#### Progress for {ex_sel}")

                c1, c2 = st.columns(2)

                with c1:
                    st.write("Weight over time (total load, lbs)")
                    chart_df = sub[["datetime", "total_weight_lbs"]].set_index("datetime")
                    st.line_chart(chart_df, use_container_width=True)

                    st.write("Volume (total load × reps, lbs·reps)")
                    vol_df = sub[["datetime", "volume"]].set_index("datetime")
                    st.line_chart(vol_df, use_container_width=True)

                with c2:
                    st.write("Difficulty trend")
                    diff_df = sub[["datetime", "difficulty"]].set_index("datetime")
                    st.line_chart(diff_df, use_container_width=True)

                    st.write("Estimated 1RM over time (lbs)")
                    est_df = sub[["datetime", "est_1rm_lbs"]].set_index("datetime")
                    st.line_chart(est_df, use_container_width=True)

                sub_display = sub.copy()
                if "user_bodyweight_kg" in sub_display.columns:
                    if unit_system.startswith("Imperial"):
                        sub_display["bodyweight_lbs"] = sub_display["user_bodyweight_kg"].apply(kg_to_lbs)
                    else:
                        sub_display["bodyweight_kg"] = sub_display["user_bodyweight_kg"]

                with st.expander("Raw log for this exercise", expanded=False):
                    st.dataframe(sub_display, use_container_width=True)

        if show_log_table:
            st.markdown("---")
            st.subheader("Full Log (tail)")
            if log_df.empty:
                st.write("No sets logged yet.")
            else:
                log_display = log_df.copy()
                if "user_bodyweight_kg" in log_display.columns:
                    if unit_system.startswith("Imperial"):
                        log_display["bodyweight_lbs"] = log_display["user_bodyweight_kg"].apply(kg_to_lbs)
                    else:
                        log_display["bodyweight_kg"] = log_display["user_bodyweight_kg"]
                st.dataframe(log_display.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()

