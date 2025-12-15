import warnings
warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited")

import io
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "scores" / "scores.db"
TASK_FILE = BASE_DIR / "data" / "full_set.jsonl"
STYLES_PATH = Path(__file__).resolve().parents[1] / "styles.css"

# Colors matching the app theme
ACCENT_PRIMARY = "#8B5CF6"
ACCENT_SECONDARY = "#7C3AED"
BG_COLOR = "#0E1117"
CARD_BG = "#1a1a2e"
TEXT_COLOR = "#FAFAFA"
TEXT_MUTED = "#9CA3AF"


def inject_css():
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def generate_comparison_graphic(by_cat_df, model1, model2):
    """Generate a butterfly/diverging bar chart comparing two models."""

    # Get scores for each model by category
    m1_scores = by_cat_df[by_cat_df["model"] == model1].set_index("category")["quality_mean"]
    m2_scores = by_cat_df[by_cat_df["model"] == model2].set_index("category")["quality_mean"]

    # Get all categories and ensure both models have data
    categories = sorted(set(m1_scores.index) | set(m2_scores.index))

    if not categories:
        return None

    # Prepare data
    m1_values = [m1_scores.get(cat, 0) for cat in categories]
    m2_values = [m2_scores.get(cat, 0) for cat in categories]

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(14, max(6, len(categories) * 0.8)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos = range(len(categories))
    bar_height = 0.6
    max_score = 5  # Quality scores are 1-5

    # Draw bars extending from center
    # Model 1 (left side) - bars go negative direction
    ax.barh(y_pos, [-v for v in m1_values], height=bar_height,
            color=ACCENT_PRIMARY, alpha=0.9, label=model1)

    # Model 2 (right side) - bars go positive direction
    ax.barh(y_pos, m2_values, height=bar_height,
            color=ACCENT_SECONDARY, alpha=0.9, label=model2)

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(m1_values, m2_values)):
        # Model 1 label (left)
        if v1 > 0:
            ax.text(-v1 - 0.15, i, f"{v1:.1f}", ha='right', va='center',
                    color=TEXT_COLOR, fontsize=11, fontweight='500')
        # Model 2 label (right)
        if v2 > 0:
            ax.text(v2 + 0.15, i, f"{v2:.1f}", ha='left', va='center',
                    color=TEXT_COLOR, fontsize=11, fontweight='500')

    # Category labels in center
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='500', color=TEXT_COLOR)

    # Style the axis
    ax.axvline(x=0, color=TEXT_MUTED, linewidth=0.8, alpha=0.5)
    ax.set_xlim(-max_score - 0.8, max_score + 0.8)
    ax.set_ylim(-0.5, len(categories) - 0.5)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove x-axis ticks
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0, pad=15)

    # Add model names as headers
    m1_short = model1.split('/')[-1] if '/' in model1 else model1
    m2_short = model2.split('/')[-1] if '/' in model2 else model2

    # Model headers at top
    ax.text(-max_score/2, len(categories) + 0.3, m1_short,
            ha='center', va='bottom', fontsize=14, fontweight='700',
            color=ACCENT_PRIMARY)
    ax.text(max_score/2, len(categories) + 0.3, m2_short,
            ha='center', va='bottom', fontsize=14, fontweight='700',
            color=ACCENT_SECONDARY)

    # Title
    fig.suptitle("Model Comparison by Category", fontsize=18, fontweight='700',
                 color=TEXT_COLOR, y=0.98)

    # Subtitle
    ax.text(0, len(categories) + 0.8, "Quality Score (1-5)",
            ha='center', va='bottom', fontsize=10, color=TEXT_MUTED)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return fig


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
    inject_css()
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

    # Get all unique models for comparison
    all_models = sorted(scores_df["model"].unique())

    render_overall(overall_df)
    render_by_category(by_cat_df)
    render_by_task(by_task_df, tasks_df, show_task_table)

    # Export comparison section
    st.markdown("---")
    st.subheader("Export Comparison Graphic")

    if len(all_models) < 2:
        st.info("Need at least 2 models with scores to generate comparison.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox("Model 1 (Left)", all_models, index=0)
        with col2:
            # Default to second model if available
            default_idx = 1 if len(all_models) > 1 else 0
            model2 = st.selectbox("Model 2 (Right)", all_models, index=default_idx)

        if model1 == model2:
            st.warning("Please select two different models to compare.")
        else:
            # Get unfiltered by_category data for the graphic
            _, full_by_cat_df, _ = compute_aggregates(scores_df, tasks_df, None)

            if st.button("Generate Comparison", type="primary"):
                with st.spinner("Generating graphic..."):
                    fig = generate_comparison_graphic(full_by_cat_df, model1, model2)

                    if fig:
                        # Display the figure
                        st.pyplot(fig)

                        # Create download button
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR, edgecolor='none')
                        buf.seek(0)

                        st.download_button(
                            label="Download PNG",
                            data=buf,
                            file_name=f"comparison_{model1.replace('/', '_')}_vs_{model2.replace('/', '_')}.png",
                            mime="image/png",
                        )
                        plt.close(fig)
                    else:
                        st.error("No category data available for these models.")


if __name__ == "__main__":
    main()
