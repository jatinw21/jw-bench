import streamlit as st
import json
import os
import random
import time
from pathlib import Path
import sqlite3

# --------------------------------------------
# CONFIG
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
TASK_FILE = BASE_DIR / "data/full_set.jsonl"
DB_PATH = BASE_DIR / "scores" / "scores.db"

# --------------------------------------------
# LOAD TASKS
# --------------------------------------------
@st.cache_data
def load_tasks():
    tasks = []
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks

# --------------------------------------------
# LOAD MODEL RESPONSES FOR A TASK
# --------------------------------------------
def load_responses(task_id):
    outputs = {}

    # e.g. outputs/openai/gpt-4o-mini/task.txt
    for vendor_dir in Path(OUTPUT_DIR).glob("*"):
        if not vendor_dir.is_dir():
            continue
        for model_dir in vendor_dir.glob("*"):
            if not model_dir.is_dir():
                continue

            file_path = model_dir / f"{task_id}.txt"
            if file_path.exists():
                outputs[f"{vendor_dir.name}/{model_dir.name}"] = file_path.read_text(encoding="utf-8")

    return outputs


# --------------------------------------------
# SQLITE HELPERS
# --------------------------------------------
def init_db():
    os.makedirs(DB_PATH.parent, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                task_id TEXT NOT NULL,
                model TEXT NOT NULL,
                quality INTEGER NOT NULL,
                tone INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (task_id, model)
            )
            """
        )

def load_scores_for_task(task_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT model, quality, tone FROM scores WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    return {model: {"quality": quality, "tone": tone} for model, quality, tone in rows}

def save_scores_for_task(task_id, scores):
    now = time.time()
    records = [
        (task_id, model, vals["quality"], vals["tone"], now)
        for model, vals in scores.items()
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO scores (task_id, model, quality, tone, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id, model)
            DO UPDATE SET
                quality = excluded.quality,
                tone = excluded.tone,
                timestamp = excluded.timestamp
            """,
            records,
        )

# --------------------------------------------
# CUSTOM MODERN CSS
# --------------------------------------------
def inject_css():
    st.markdown("""
<style>

    .hidden-chip {
        background: #DDE3ED;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: #555;
        display: inline-block;
        margin-bottom: 10px;
    }

    .reveal-chip {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: white;
        display: inline-block;
        margin-bottom: 10px;
    }

</style>
    """, unsafe_allow_html=True)

# --------------------------------------------
# COLOR MAPPING FOR MODELS
# --------------------------------------------
COLOR_MAP = [
    "#377DFF",  # Blue
    "#A259FF",  # Purple
    "#2EB67D",  # Green
    "#FF7A59",  # Orange
]

def model_color(idx):
    return COLOR_MAP[idx % len(COLOR_MAP)]

# --------------------------------------------
# MAIN UI
# --------------------------------------------
def main():
    st.set_page_config(page_title="LLM Model Scoring UI", layout="wide")
    init_db()
    inject_css()

    tasks = load_tasks()
    task_ids = [t["id"] for t in tasks]
    params = st.query_params
    # Extract task from query params (supports str or list)
    param_task = params.get("task")
    if isinstance(param_task, list):
        param_task = param_task[0] if param_task else None
    if param_task not in task_ids:
        param_task = task_ids[0]

    #-----------------------------------------
    # SIDEBAR
    #-----------------------------------------
    st.sidebar.title("Controls")
    selected_task_id = st.sidebar.selectbox(
        "Select Task",
        task_ids,
        index=task_ids.index(param_task),
    )
    # Keep query params in sync with selection
    if selected_task_id != param_task:
        st.query_params["task"] = selected_task_id

    curr_index = task_ids.index(selected_task_id)
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("< Previous") and curr_index > 0:
        st.query_params["task"] = task_ids[curr_index - 1]

    if col_next.button("Next >") and curr_index < len(task_ids) - 1:
        st.query_params["task"] = task_ids[curr_index + 1]

    st.sidebar.markdown("---")
    st.sidebar.write("Autosave: Enabled")

    #-----------------------------------------
    # MAIN AREA
    #-----------------------------------------
    task = next(t for t in tasks if t["id"] == selected_task_id)
    saved_scores = load_scores_for_task(selected_task_id)

    st.markdown(f"### {task['id']}  \n**Category:** {task['category']}")
    st.markdown(f"**Prompt:**  \n{task['prompt']}")

    # Load model outputs
    responses = load_responses(selected_task_id)
    model_names = list(responses.keys())

    # Randomize order per session per task
    seed_key = f"shuffle_{selected_task_id}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.random()
    random.Random(st.session_state[seed_key]).shuffle(model_names)

    # Detect if all scored → reveal names
    reveal_key = f"reveal_{selected_task_id}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    st.markdown("### Model Responses")
    print(model_names)
    cols = st.columns(len(model_names))

    for idx, model in enumerate(model_names):
        with cols[idx]:
            # Hidden name or revealed name
            if not st.session_state[reveal_key]:
                st.markdown(f"<span class='hidden-chip'>Model {chr(65+idx)}</span>", unsafe_allow_html=True)
            else:
                chip_color = model_color(idx)
                st.markdown(
                    f"<span class='reveal-chip' style='background:{chip_color}'>{model}</span>",
                    unsafe_allow_html=True
                )

            # Response text
            with st.expander("View Response", expanded=True):
                st.write(responses[model])

            # Sliders
            q_key = f"{selected_task_id}_q_{model}"
            t_key = f"{selected_task_id}_t_{model}"

            if q_key not in st.session_state and model in saved_scores:
                st.session_state[q_key] = saved_scores[model]["quality"]
            if t_key not in st.session_state and model in saved_scores:
                st.session_state[t_key] = saved_scores[model]["tone"]

            quality = st.slider(
                f"Quality (Model {chr(65+idx)})",
                1,
                5,
                st.session_state.get(q_key, saved_scores.get(model, {}).get("quality", 3)),
                key=q_key,
            )
            tone = st.slider(
                f"Tone Fit (Model {chr(65+idx)})",
                1,
                5,
                st.session_state.get(t_key, saved_scores.get(model, {}).get("tone", 3)),
                key=t_key,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # SAVE BUTTON
    if model_names:
        if st.button("Save scores for this task"):
            score_payload = {}
            for model in model_names:
                q_key = f"{selected_task_id}_q_{model}"
                t_key = f"{selected_task_id}_t_{model}"
                score_payload[model] = {
                    "quality": int(st.session_state.get(q_key, 3)),
                    "tone": int(st.session_state.get(t_key, 3)),
                }
            save_scores_for_task(selected_task_id, score_payload)
            st.success(f"Saved scores for {selected_task_id}")
            saved_scores = load_scores_for_task(selected_task_id)

    # REVEAL BUTTON
    all_scored = all(
        m in saved_scores
        for m in model_names
    )

    if not st.session_state[reveal_key] and all_scored:
        if st.button("Reveal Model Names"):
            time.sleep(0.2)
            st.session_state[reveal_key] = True
            st.experimental_rerun()


# --------------------------------------------
if __name__ == "__main__":
    main()
