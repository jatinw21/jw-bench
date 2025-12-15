# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A subjective, preference-aligned LLM benchmarking system. It runs models on curated tasks across seven categories (Fun, Insightful, BS-Meter, Teaching, Professional, Planning, Critique), stores outputs, and provides a Streamlit interface for blind human scoring on a Quality axis.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then set OPENROUTER_API_KEY

# Generate model outputs
python run.py                    # Run all tasks for configured models
python -c "from run import main; main(sample=True)"  # Run sample subset only

# Verify all outputs exist
python verify_outputs.py

# Launch scoring UI
streamlit run ui/scoring_app.py

# Launch standalone leaderboard
streamlit run ui/leaderboard_app.py
```

## Architecture

**Data Flow**: Tasks (JSONL) → Model API calls → Output files → Blind scoring UI → SQLite scores → Leaderboard

**Key Files**:
- `run.py` - Orchestrates model runs; configure `MODELS` list to add/remove models
- `client_openrouter.py` - OpenRouter API client wrapper
- `task_loader.py` - Loads tasks from `data/full_set.jsonl`; `SAMPLE_IDS` for quick testing
- `ui/scoring_app.py` - Main Streamlit app with blind scoring (models shown as A/B/C until reveal)
- `ui/pages/leaderboard.py` - Multi-page leaderboard accessible from scoring app

**Storage**:
- `data/full_set.jsonl` - Task definitions with `id`, `category`, `prompt` fields
- `outputs/{vendor}/{model}/{task_id}.txt` - Model responses (nested by vendor/model)
- `scores/scores.db` - SQLite database with `(task_id, model, quality, timestamp)` scores

## Key Patterns

- Models are specified as OpenRouter paths: `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`
- Output directory structure matches model path: `outputs/openai/gpt-4o-mini/task.txt`
- Scoring UI randomizes model display order per task (shuffled by session seed)
- Task completion requires scoring all models for a task before advancing