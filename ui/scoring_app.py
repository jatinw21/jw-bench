import streamlit as st
import json
import os
import random
import time
import html
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

def task_completion_counts(task_ids, expected_models):
    if expected_models == 0:
        return 0, len(task_ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT task_id, COUNT(*) FROM scores GROUP BY task_id"
        ).fetchall()
    counts = {task: count for task, count in rows}
    completed = sum(1 for t in task_ids if counts.get(t, 0) >= expected_models)
    return completed, len(task_ids)

# --------------------------------------------
# CUSTOM MODERN CSS
# --------------------------------------------
def inject_css():
    st.markdown("""
<style>
    .page-shell { padding-top: 12px; }
    .page-title { margin: 4px 0 8px 0; }
    .section-heading { margin: 12px 0 8px 0; }

    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 0 4px 0;
        flex-wrap: wrap;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .pill-muted { background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.85); }
    .pill-strong { background: linear-gradient(120deg, #6E8EF5, #9B6BFF); color: #fff; }
    .pill-soft { background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.75); }

    .prompt-card, .model-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 18px;
    }
    .model-card { height: 100%; min-height: 480px; display: flex; flex-direction: column; gap: 12px; }

    .card-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.7);
        margin-bottom: 6px;
    }

    .response-box {
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 14px;
        color: rgba(255,255,255,0.92);
        font-size: 14px;
        line-height: 1.55;
        max-height: 320px;
        overflow: auto;
        white-space: pre-wrap;
    }

    .hidden-chip {
        background: rgba(255,255,255,0.12);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: rgba(255,255,255,0.85);
        display: inline-block;
        margin-bottom: 6px;
    }

    .reveal-chip {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: white;
        display: inline-block;
        margin-bottom: 6px;
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
    # Extract category from query params
    categories = sorted({t["category"] for t in tasks})
    category_options = ["All"] + categories
    param_category = params.get("category")
    if isinstance(param_category, list):
        param_category = param_category[0] if param_category else None
    if param_category not in category_options:
        param_category = "All"

    #-----------------------------------------
    # SIDEBAR
    #-----------------------------------------
    st.sidebar.title("Controls")
    selected_category = st.sidebar.selectbox(
        "Select Category",
        category_options,
        index=category_options.index(param_category),
    )
    if selected_category != param_category:
        st.query_params["category"] = selected_category

    filtered_tasks = [t for t in tasks if selected_category == "All" or t["category"] == selected_category]
    filtered_task_ids = [t["id"] for t in filtered_tasks]
    if param_task not in filtered_task_ids:
        param_task = filtered_task_ids[0]

    selected_task_id = st.sidebar.selectbox(
        "Select Task",
        filtered_task_ids,
        index=filtered_task_ids.index(param_task),
    )
    # Keep query params in sync with selection
    if selected_task_id != param_task:
        st.query_params["task"] = selected_task_id
        st.query_params["category"] = selected_category

    st.sidebar.caption("Use the header arrows to move between tasks.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Scores are saved to scores/scores.db")

    #-----------------------------------------
    # MAIN AREA
    #-----------------------------------------
    task = next(t for t in tasks if t["id"] == selected_task_id)
    saved_scores = load_scores_for_task(selected_task_id)

    # Load model outputs
    responses = load_responses(selected_task_id)
    model_names = list(responses.keys())
    expected_models = len(model_names)
    scored_models = len(saved_scores)
    completed_tasks, total_tasks = task_completion_counts(task_ids, expected_models or 1)

    # Navigation indices within filtered set
    curr_index = filtered_task_ids.index(selected_task_id)

    # Top bar with chips, progress, navigation
    st.markdown("<div class='page-shell'>", unsafe_allow_html=True)
    with st.container():
        top_left, top_mid, top_right = st.columns([3, 3, 2])
        with top_left:
            st.markdown(f"<div class='pill pill-muted'>{task['category']}</div>", unsafe_allow_html=True)
            st.markdown(f"<h1 class='page-title'>{task['id']}</h1>", unsafe_allow_html=True)
        with top_mid:
            st.markdown(
                f"<div class='pill pill-strong'>{min(scored_models, expected_models)}/{expected_models or '0'} models scored</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='pill pill-soft'>{completed_tasks}/{total_tasks} tasks fully scored</div>",
                unsafe_allow_html=True,
            )
        with top_right:
            nav_prev, nav_next = st.columns(2)
            if filtered_task_ids:
                if nav_prev.button("← Previous", use_container_width=True, disabled=curr_index == 0):
                    new_idx = max(0, curr_index - 1)
                    st.query_params["task"] = filtered_task_ids[new_idx]
                    st.query_params["category"] = selected_category
                if nav_next.button("Next →", use_container_width=True, disabled=curr_index >= len(filtered_task_ids) - 1):
                    new_idx = min(len(filtered_task_ids) - 1, curr_index + 1)
                    st.query_params["task"] = filtered_task_ids[new_idx]
                    st.query_params["category"] = selected_category

    # Prompt card
    st.markdown(
        f"""
        <div class='prompt-card'>
            <div class='card-label'>Prompt</div>
            <div>{task['prompt']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Randomize order per session per task
    seed_key = f"shuffle_{selected_task_id}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.random()
    random.Random(st.session_state[seed_key]).shuffle(model_names)

    # Detect if all scored → reveal names
    reveal_key = f"reveal_{selected_task_id}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    st.markdown("### Model Responses", unsafe_allow_html=True)
    save_clicked = False

    if not model_names:
        st.info("No model outputs found for this task.")
    else:
        # Header actions
        if st.button("💾 Save scores", type="primary"):
            save_clicked = True

        cols_per_row = 2 if len(model_names) > 1 else 1
        for start in range(0, len(model_names), cols_per_row):
            row_models = model_names[start:start + cols_per_row]
            cols = st.columns(len(row_models))
            for idx_in_row, model in enumerate(row_models):
                idx = start + idx_in_row
                with cols[idx_in_row]:
                    chip_html = (
                        f"<span class='hidden-chip'>Model {chr(65+idx)}</span>"
                        if not st.session_state[reveal_key]
                        else f"<span class='reveal-chip' style='background:{model_color(idx)}'>{model}</span>"
                    )
                    st.markdown(chip_html, unsafe_allow_html=True)

                    # Response box
                    st.markdown(
                        f"""
                        <div class='model-card'>
                            <div class='card-label'>Response</div>
                            <div class='response-box'>{html.escape(responses[model])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Sliders
                    q_key = f"{selected_task_id}_q_{model}"
                    t_key = f"{selected_task_id}_t_{model}"

                    if q_key not in st.session_state and model in saved_scores:
                        st.session_state[q_key] = saved_scores[model]["quality"]
                    if t_key not in st.session_state and model in saved_scores:
                        st.session_state[t_key] = saved_scores[model]["tone"]

                    st.slider(
                        "Quality",
                        1,
                        5,
                        st.session_state.get(q_key, saved_scores.get(model, {}).get("quality", 3)),
                        key=q_key,
                    )
                    st.slider(
                        "Tone Fit",
                        1,
                        5,
                        st.session_state.get(t_key, saved_scores.get(model, {}).get("tone", 3)),
                        key=t_key,
                    )

    # SAVE HANDLER
    if save_clicked and model_names:
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
