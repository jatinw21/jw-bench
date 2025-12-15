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
def get_all_available_models():
    """Get all model names from outputs directory."""
    models = []
    for vendor_dir in Path(OUTPUT_DIR).glob("*"):
        if not vendor_dir.is_dir():
            continue
        for model_dir in vendor_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            models.append(f"{vendor_dir.name}/{model_dir.name}")
    return sorted(models)


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
                timestamp REAL NOT NULL,
                PRIMARY KEY (task_id, model)
            )
            """
        )
        # Migrate old schema with tone column to new schema without tone
        cols = [row[1] for row in conn.execute("PRAGMA table_info(scores)").fetchall()]
        if "tone" in cols:
            conn.execute("ALTER TABLE scores RENAME TO scores_old")
            conn.execute(
                """
                CREATE TABLE scores (
                    task_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    quality INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (task_id, model)
                )
                """
            )
            conn.execute(
                "INSERT INTO scores (task_id, model, quality, timestamp) SELECT task_id, model, quality, timestamp FROM scores_old"
            )
            conn.execute("DROP TABLE scores_old")

def load_scores_for_task(task_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT model, quality FROM scores WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    return {model: {"quality": quality} for model, quality in rows}

def save_scores_for_task(task_id, scores):
    now = time.time()
    records = [
        (task_id, model, vals["quality"], now)
        for model, vals in scores.items()
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO scores (task_id, model, quality, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id, model)
            DO UPDATE SET
                quality = excluded.quality,
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


def clamp_task(active_task, filtered_task_ids, direction):
    """
    Given current task and direction (-1 for prev, 1 for next), return clamped target task and index.
    """
    if not filtered_task_ids:
        return None, None
    try:
        idx = filtered_task_ids.index(active_task)
    except ValueError:
        idx = 0
    new_idx = min(max(idx + direction, 0), len(filtered_task_ids) - 1)
    return filtered_task_ids[new_idx], new_idx


def parse_query_params(task_ids, category_options):
    params = st.query_params
    param_task = params.get("task")
    if isinstance(param_task, list):
        param_task = param_task[0] if param_task else None
    param_category = params.get("category")
    if isinstance(param_category, list):
        param_category = param_category[0] if param_category else None
    if param_category not in category_options:
        param_category = "All"
    if param_task not in task_ids:
        param_task = task_ids[0]
    return param_task, param_category


def ensure_session_defaults(param_task, param_category, task_ids):
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = param_category
    if "active_task" not in st.session_state:
        st.session_state.active_task = param_task or task_ids[0]


def render_sidebar(tasks, category_options, all_models):
    st.sidebar.markdown("<div style='font-size:14px;'>", unsafe_allow_html=True)
    st.sidebar.title("Controls")
    selected_category = st.sidebar.selectbox(
        "Select Category",
        category_options,
        index=category_options.index(st.session_state.selected_category),
    )
    if selected_category != st.session_state.selected_category:
        st.session_state.selected_category = selected_category
        st.query_params["category"] = selected_category
        st.session_state.active_task = None
        st.rerun()

    filtered_tasks = [t for t in tasks if selected_category == "All" or t["category"] == selected_category]
    filtered_task_ids = [t["id"] for t in filtered_tasks]
    if not filtered_task_ids:
        st.error("No tasks for this category.")
        return selected_category, None, [], []

    if st.session_state.active_task not in filtered_task_ids:
        st.session_state.active_task = filtered_task_ids[0]
        st.query_params["task"] = st.session_state.active_task
        st.query_params["category"] = selected_category

    selected_task_id = st.sidebar.selectbox(
        "Select Task",
        filtered_task_ids,
        index=filtered_task_ids.index(st.session_state.active_task),
    )
    if selected_task_id != st.session_state.active_task:
        st.session_state.active_task = selected_task_id
        st.query_params["task"] = selected_task_id
        st.query_params["category"] = selected_category
        st.rerun()

    st.sidebar.markdown("---")

    # Model selection
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = all_models

    selected_models = st.sidebar.multiselect(
        "Models to Compare",
        options=all_models,
        default=st.session_state.selected_models,
        help="Select which models to display and score",
    )

    if selected_models != st.session_state.selected_models:
        st.session_state.selected_models = selected_models
        st.rerun()

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    return selected_category, selected_task_id, filtered_task_ids, selected_models


def render_topbar(task, scored_models, expected_models, completed_tasks, total_tasks, filtered_task_ids, selected_category):
    curr_index = filtered_task_ids.index(task["id"])
    st.markdown("<div class='page-shell'>", unsafe_allow_html=True)
    with st.container():
        top_left, top_mid, top_right = st.columns([3, 3, 2])
        with top_left:
            st.markdown(f"<div class='pill pill-muted'>{task['category']} › {task['id']}</div>", unsafe_allow_html=True)
        with top_mid:
            task_scored = expected_models > 0 and scored_models >= expected_models
            status_text = "Task scored" if task_scored else "Task not scored"
            pill_class = "pill-strong" if task_scored else "pill-muted"
            st.markdown(
                f"<div style='display:flex; gap:8px; align-items:center; flex-wrap:wrap;'>"
                f"<span class='pill {pill_class}'>{status_text}</span>"
                f"<span class='pill pill-soft'>{completed_tasks}/{total_tasks} tasks fully scored</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with top_right:
            nav_prev, nav_next = st.columns(2)
            if filtered_task_ids:
                if nav_prev.button("← Previous", use_container_width=True, disabled=curr_index == 0):
                    target_task, _ = clamp_task(task["id"], filtered_task_ids, -1)
                    st.session_state.active_task = target_task
                    st.query_params["task"] = target_task
                    st.query_params["category"] = selected_category
                    st.rerun()
                if nav_next.button("Next →", use_container_width=True, disabled=curr_index >= len(filtered_task_ids) - 1):
                    target_task, _ = clamp_task(task["id"], filtered_task_ids, 1)
                    st.session_state.active_task = target_task
                    st.query_params["task"] = target_task
                    st.query_params["category"] = selected_category
                    st.rerun()


def render_prompt(task):
    st.markdown(
        f"""
        <div class='prompt-card' style="margin: 36px 0;">
            <div class='card-label'>Prompt</div>
            <div style="font-size:18px; margin-top:12px;">{task['prompt']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_responses(task_id, model_names, responses, saved_scores):
    # Randomize order per session per task
    seed_key = f"shuffle_{task_id}"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.random()
    random.Random(st.session_state[seed_key]).shuffle(model_names)

    # Detect if all scored → reveal names
    reveal_key = f"reveal_{task_id}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    save_clicked = False

    if not model_names:
        st.info("No model outputs found for this task.")
        return saved_scores

    cols_per_row = 2 if len(model_names) > 1 else 1
    for start in range(0, len(model_names), cols_per_row):
        row_models = model_names[start:start + cols_per_row]
        cols = st.columns(len(row_models))
        for idx_in_row, model in enumerate(row_models):
            idx = start + idx_in_row
            with cols[idx_in_row]:
                label = f"Model {chr(65+idx)} response" if not st.session_state[reveal_key] else f"{model} response"
                body_md = responses[model].replace("\n", "<br>")
                wrapped = (
                    f"<div class='model-card'>"
                    f"<div class='card-label'>{label}</div>"
                    f"<div class='response-box'>\n\n{body_md}\n\n</div>"
                    f"</div>"
                )
                st.markdown(wrapped, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    # Scoring sliders under cards (aligned heights)
    cols_per_row = 2 if len(model_names) > 1 else 1
    for start in range(0, len(model_names), cols_per_row):
        row_models = model_names[start:start + cols_per_row]
        cols = st.columns(len(row_models))
        for idx_in_row, model in enumerate(row_models):
            idx = start + idx_in_row
            with cols[idx_in_row]:
                sub_left, sub_mid, sub_right = st.columns([0.1, 0.8, 0.1])
                with sub_mid:
                    q_key = f"{task_id}_q_{model}"
                    st.slider(
                        f"Quality ({'Model ' + chr(65+idx) if not st.session_state[reveal_key] else model})",
                        1,
                        5,
                        st.session_state.get(q_key, saved_scores.get(model, {}).get("quality", 3)),
                        key=q_key,
                    )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    spacer_left, center_col, spacer_right = st.columns([2, 3, 2])
    with center_col:
        if st.button("💾 Save scores", type="primary", use_container_width=True):
            save_clicked = True
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    spacer_left2, center_col2, spacer_right2 = st.columns([2, 3, 2])
    with center_col2:
        all_scored = all(m in saved_scores for m in model_names)
        reveal_disabled = not all_scored
        if st.button("Reveal Model Names", disabled=reveal_disabled, use_container_width=True):
            st.session_state[f"reveal_{task_id}"] = True

    if save_clicked and model_names:
        score_payload = {}
        for model in model_names:
            q_key = f"{task_id}_q_{model}"
            score_payload[model] = {
                "quality": int(st.session_state.get(q_key, 3)),
            }
        save_scores_for_task(task_id, score_payload)
        st.rerun()

    return saved_scores

# --------------------------------------------
# CUSTOM MODERN CSS
# --------------------------------------------
STYLES_PATH = Path(__file__).parent / "styles.css"

def inject_css():
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

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
    categories = sorted({t["category"] for t in tasks})
    category_options = ["All"] + categories
    all_models = get_all_available_models()

    param_task, param_category = parse_query_params(task_ids, category_options)
    ensure_session_defaults(param_task, param_category, task_ids)

    selected_category, selected_task_id, filtered_task_ids, selected_models = render_sidebar(tasks, category_options, all_models)
    if not selected_task_id:
        return

    task = next(t for t in tasks if t["id"] == selected_task_id)
    saved_scores = load_scores_for_task(selected_task_id)

    responses = load_responses(selected_task_id)
    # Filter to only selected models
    model_names = [m for m in responses.keys() if m in selected_models]
    expected_models = len(model_names)
    # Only count scores for selected models
    scored_models = sum(1 for m in model_names if m in saved_scores)
    completed_tasks, total_tasks = task_completion_counts(task_ids, expected_models or 1)

    render_topbar(task, scored_models, expected_models, completed_tasks, total_tasks, filtered_task_ids, selected_category)
    spacer_left, prompt_col, spacer_right = st.columns([0.15, 0.7, 0.15])
    with prompt_col:
        render_prompt(task)
    render_model_responses(selected_task_id, model_names, responses, saved_scores)


# --------------------------------------------
if __name__ == "__main__":
    main()
