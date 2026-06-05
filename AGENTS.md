# Agent guide

Orientation for AI coding agents working in this repo. Humans should read `README.md` and `DEPLOYMENT.md` first.

## What this project is

A demo built around Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) book. The model code (transformer, attention, training loop, generation) is adapted from that repo; this project wraps it in CLI scripts and a Flask web UI for demoing.

Reference checkout (already present locally): `~/projects/LLMs-from-scratch`. Consult it for the original chapter code, datasets, and explanations when behavior here is unclear or you need to trace something back to the source. Do **not** edit that repo from this project — it is an upstream reference.

## Layout

- `llm_demo/` — model + training + generation, lifted/adapted from the book.
  - `model.py` — `GPTModel`, `TransformerBlock`, `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `GPTDatasetV1`, `create_dataloader_v1`. Mirrors chapters 2–4.
  - `training.py` — pretraining loop.
  - `generation.py` — text generation / sampling.
  - `config.py` — `MODEL_PRESETS` (`demo-small`, `124M`), `DEFAULT_TRAINING_SETTINGS`, `BOOK_TEXT_PATH` (points into the LLMs-from-scratch checkout for "The Verdict").
- `scripts/` — CLI entry points: `random_demo.py`, `pretrain.py`, `generate.py`.
- `app/` — Flask web UI that loads a trained model and serves completions.
  - `__init__.py` — app factory; requires `MODEL_WEIGHTS_PATH` env var unless `LOAD_MODEL=False` (test mode).
  - `routes/`, `services/`, `templates/`, `static/`, `config.py`.
- `tests/` — pytest suite (`test_llm_demo.py`, `test_web_app.py`).
- `models/` — local model weight files (gitignored).
- `texts/` — local training corpora.
- `Dockerfile`, `compose.yml`, `compose.dev.yml`, `compose.prod.yml`, `Makefile` — container + deployment plumbing. See `DEPLOYMENT.md`.

## Conventions

- Python is run via the project virtualenv: `.venv/bin/python`. The Makefile auto-detects this.
- Run tests with `make test` (or `.venv/bin/python -m pytest tests`).
- Run the web UI locally with `make flask-run`; via Docker with `make run` / `make dev`.
- `.env` must exist before any `make run`/`make dev`/`make docker-test` (copy from `.env.example`).
- Default model preset is `124M`. `demo-small` is a tiny shape useful for fast tests / sanity checks.
- Web app: `MODEL_WEIGHTS_PATH` is **required**; set `MODEL_PRESET` to match the weights' shape.

## Working with the book code

When extending model/training/generation logic:

1. Check if the book already does it in `~/projects/LLMs-from-scratch/chXX/`. Prefer adapting the book's approach over inventing a new one, so the demo stays a faithful illustration of the source material.
2. Keep `llm_demo/` close to the book's structure and naming — readers may be cross-referencing.
3. Don't pull in heavy abstractions; the value of this code is that it stays readable next to the book.

## Deployment

Production runs on a VPS at `gpt.pdlawson.com` behind a shared `web` reverse-proxy network. Image is built by GitHub Actions on push to `main` and deployed via SSH. Full details in `DEPLOYMENT.md`.
