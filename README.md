## Why?

This project builds a **subjective, preference-aligned LLM benchmarking system** designed to evaluate models on the dimensions that actually matter for real creative and professional use, not math puzzles or synthetic benchmarks. 

It runs multiple models on a curated set of tasks across seven categories
- Fun
- Insightful
- BS-Meter
- Teaching
- Professional
- Planning
- Critique

Stores their outputs, and provides a modern Streamlit interface for blind human scoring along two axes: **Quality/Usefulness** and **Tone Fit**. 

The goal is to create a personalized, human-centric benchmark that highlights meaningful differences between models, helping determine which ones perform best for your actual workflows and taste.

## Setup
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and set your key.


## How to use
Install deps from `requirements.txt`, set `OPENROUTER_API_KEY` in `.env`.
Generate model outputs with `python run.py` (or main(sample=True) for the sample IDs).
Launch the scorer with `streamlit run ui/scoring_app.py`; review outputs, score quality/tone, then reveal identities

Outputs: `results.csv` and a console summary (accuracy by model).