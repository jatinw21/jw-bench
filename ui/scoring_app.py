import streamlit as st
import json
import os
import random
import time
from pathlib import Path

# --------------------------------------------
# CONFIG
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
TASK_FILE = BASE_DIR / "data/full_set.jsonl"
SCORES_FILE = BASE_DIR / "scores/scores.jsonl"

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
# SAVE SCORES
# --------------------------------------------
def save_score(record):
    os.makedirs("scores", exist_ok=True)
    with open(SCORES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# --------------------------------------------
# CUSTOM MODERN CSS
# --------------------------------------------
def inject_css():
    st.markdown("""
<style>

    /* Global background */
    body {
        background-color: #F7F9FC;
    }

    /* Task card */
    .task-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        border: 1px solid #EDF1F7;
    }

    /* Model card */
    .model-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #E1E8F5;
        margin-bottom: 25px;
    }

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
    inject_css()

    tasks = load_tasks()
    task_ids = [t["id"] for t in tasks]

    #-----------------------------------------
    # SIDEBAR
    #-----------------------------------------
    st.sidebar.title("Controls")
    selected_task_id = st.sidebar.selectbox("Select Task", task_ids)

    curr_index = task_ids.index(selected_task_id)
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("< Previous") and curr_index > 0:
        st.experimental_set_query_params(task=task_ids[curr_index - 1])

    if col_next.button("Next >") and curr_index < len(task_ids) - 1:
        st.experimental_set_query_params(task=task_ids[curr_index + 1])

    st.sidebar.markdown("---")
    st.sidebar.write("Autosave: Enabled")

    #-----------------------------------------
    # MAIN AREA
    #-----------------------------------------
    task = next(t for t in tasks if t["id"] == selected_task_id)

    st.markdown(f"<div class='task-card'>", unsafe_allow_html=True)
    st.markdown(f"### {task['id']}  \n**Category:** {task['category']}")
    st.markdown(f"**Prompt:**  \n{task['prompt']}")
    st.markdown("</div>", unsafe_allow_html=True)

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
            st.markdown("<div class='model-card'>", unsafe_allow_html=True)

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
            quality = st.slider(f"Quality (Model {chr(65+idx)})", 1, 5, 3, key=f"{selected_task_id}_q_{model}")
            tone = st.slider(f"Tone Fit (Model {chr(65+idx)})", 1, 5, 3, key=f"{selected_task_id}_t_{model}")

            # Autosave after scoring
            if quality and tone:
                record = {
                    "task_id": selected_task_id,
                    "model": model,
                    "quality": quality,
                    "tone": tone,
                    "timestamp": time.time()
                }
                save_score(record)

            st.markdown("</div>", unsafe_allow_html=True)

    # REVEAL BUTTON
    all_scored = all(
        st.session_state.get(f"{selected_task_id}_q_{m}") and
        st.session_state.get(f"{selected_task_id}_t_{m}")
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
