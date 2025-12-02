import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "scores" / "scores.db"
TASK_FILE = BASE_DIR / "data" / "full_set.jsonl"


@st.cache_data
def load_tasks():
    rows = []
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    tasks_df = pd.DataFrame(rows)
    return tasks_df


@st.cache_data
def load_scores():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["task_id", "model", "quality", "timestamp"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT task_id, model, quality, timestamp FROM scores", conn)
    return df


def compute_aggregates(scores_df, tasks_df, selected_category=None):
    merged = scores_df.merge(
        tasks_df[["id", "category"]],
        left_on="task_id",
        right_on="id",
        how="left",
    ).rename(columns={"id": "task_id_task"})

    if selected_category and selected_category != "All":
        merged = merged[merged["category"] == selected_category]

    overall = (
        merged.groupby("model")
        .agg(
            quality_mean=("quality", "mean"),
            quality_median=("quality", "median"),
            count=("quality", "count"),
        )
        .reset_index()
        .sort_values("quality_mean", ascending=False)
    )

    by_category = (
        merged.groupby(["category", "model"])
        .agg(quality_mean=("quality", "mean"), count=("quality", "count"))
        .reset_index()
        .sort_values(["category", "quality_mean"], ascending=[True, False])
    )

    by_task = (
        merged.groupby(["task_id", "model"])
        .agg(quality_mean=("quality", "mean"), count=("quality", "count"))
        .reset_index()
    )

    return overall, by_category, by_task


def render_overall(overall_df):
    st.subheader("Overall by Model")
    if overall_df.empty:
        st.info("No scores yet.")
        return
    st.dataframe(
        overall_df.assign(
            quality_mean=lambda d: d["quality_mean"].round(2),
            quality_median=lambda d: d["quality_median"].round(2),
        ),
        use_container_width=True,
    )


def render_by_category(by_cat_df):
    st.subheader("By Category")
    if by_cat_df.empty:
        st.info("No scores for this category filter.")
        return
    st.dataframe(
        by_cat_df.assign(quality_mean=lambda d: d["quality_mean"].round(2)),
        use_container_width=True,
    )


def render_by_task(by_task_df, tasks_df, show_table):
    if not show_table:
        return
    st.subheader("By Task (filtered)")
    if by_task_df.empty:
        st.info("No task-level scores for this filter.")
        return
    task_titles = tasks_df.set_index("id")["prompt"].to_dict()
    by_task_df = by_task_df.assign(
        prompt=lambda d: d["task_id"].map(task_titles).fillna("")
    )[
        ["task_id", "prompt", "model", "quality_mean", "count"]
    ].rename(columns={"quality_mean": "quality_mean_rounded"})
    by_task_df["quality_mean_rounded"] = by_task_df["quality_mean_rounded"].round(2)
    st.dataframe(by_task_df, use_container_width=True, height=400)


def main():
    st.set_page_config(page_title="LLM Leaderboard", layout="wide")
    st.title("LLM Scoring Leaderboard")
    st.caption("Compare model quality scores overall, by category, and by task.")

    tasks_df = load_tasks()
    scores_df = load_scores()

    categories = ["All"] + sorted(tasks_df["category"].unique())

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_category = st.sidebar.selectbox("Category", categories)
    show_task_table = st.sidebar.checkbox("Show task-level table", value=False)

    if scores_df.empty:
        st.warning("No scores found. Run the scoring page first.")
        return

    overall_df, by_cat_df, by_task_df = compute_aggregates(scores_df, tasks_df, selected_category)

    render_overall(overall_df)
    render_by_category(by_cat_df)
    render_by_task(by_task_df, tasks_df, show_task_table)


if __name__ == "__main__":
    main()
