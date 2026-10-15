# Agent guide

Orientation for AI coding agents working in this repo. Humans should read `README.md` and `DEPLOYMENT.md` first.

## What this project is

A demo built around Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) book. The model code (transformer, attention, training loop, generation) is adapted from that repo; this project wraps it in CLI scripts and a Flask web UI for demoing.

Reference checkout (already present locally): `~/projects/LLMs-from-scratch`. Consult it for the original chapter code, datasets, and explanations when behavior here is unclear or you need to trace something back to the source. Do **not** edit that repo from this project — it is an upstream reference.

## Layout

- `llm/` — model + training + generation, lifted/adapted from the book. (Reusable model code, independent of any demo.)
  - `model.py` — `GPTModel`, `TransformerBlock`, `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `GPTDatasetV1`, `create_dataloader_v1`. Mirrors chapters 2–4.
  - `training.py` — pretraining loop.
  - `generation.py` — text generation / sampling.
  - `config.py` — `MODEL_PRESETS` (from-scratch: `demo-small`, `124M`; GPT-2 open weights: `gpt2-small`/`-medium`/`-large`/`-xl`), `DEFAULT_TRAINING_SETTINGS`, `BOOK_TEXT_PATH` (points into the LLMs-from-scratch checkout for "The Verdict").
- `scripts/` — CLI entry points: `random_demo.py`, `pretrain.py`, `generate.py`, `fetch_gpt2.py` (one-off, dev-only: downloads OpenAI's GPT-2 weights as a PyTorch state dict — pure torch + requests, no TensorFlow — and saves `weights/gpt2-small.pth`).
- `app/` — Flask web UI hosting multiple demos: a landing page plus one page per demo.
  - `demos.py` — `Demo` registry (slug, title, `task`, preset, weights filename, copy). A demo is shown only if its weights file exists in `WEIGHTS_DIR`.
  - `services/registry.py` — `DemoRegistry`: lazily builds + caches one runner per demo (loads each model on first request).
  - `services/completer.py` — `LLMCompleter`, the `completion`-task runner.
  - `__init__.py` — app factory; builds the registry from `WEIGHTS_DIR`.
  - `routes/` (`/` landing, `/<slug>` per demo), `templates/` (`landing.html`, `completion.html`), `static/`, `config.py`.
- `tests/` — pytest suite (`test_llm.py`, `test_web_app.py`).
- `weights/` — local model weight files, one `.pth` per demo (gitignored).
- `texts/` — local training corpora.
- `Dockerfile`, `compose.yml`, `compose.prod.yml`, `Makefile` — container + deployment plumbing. See `DEPLOYMENT.md`.

## Conventions

- Python is run via the worktree's `.venv/bin/python`; setup is `make setup`.
- Run tests with `make test`; start the web UI with `make dev-start`.
- Default model preset is `124M`. `demo-small` is a tiny shape useful for fast tests / sanity checks. The `gpt2-*` presets match OpenAI's open weights (1024 context, `qkv_bias=True`, no dropout) — same `GPTModel`, different config.
- Web app: set `WEIGHTS_DIR` to the directory of `.pth` files. Demos self-register from `app/demos.py` and appear only when their weights file is present — no per-demo env vars needed.
- Apple GPU / MPS note: on this Mac, `.venv` PyTorch has MPS built in and MPS works outside the Codex sandbox, but sandboxed checks may report `torch.backends.mps.is_available() == False`. For real training or generation runs that should use the Apple GPU, run the same `.venv/bin/python ... --device mps` command with escalation rather than falling back to CPU. Use unbuffered Python (`.venv/bin/python -u ...`) for long training runs so progress logs appear live.

## Working with the book code

When extending model/training/generation logic:

1. Check if the book already does it in `~/projects/LLMs-from-scratch/chXX/`. Prefer adapting the book's approach over inventing a new one, so the demo stays a faithful illustration of the source material.
2. Keep `llm/` close to the book's structure and naming — readers may be cross-referencing.
3. Don't pull in heavy abstractions; the value of this code is that it stays readable next to the book.

## Deployment

Production runs on a VPS at `gpt.pdlawson.com` behind a shared `web`
reverse-proxy network. GitHub Actions builds and publishes an image on pushes to
`main`, then the VPS independently pulls and deploys changed `main` digests from
a root-owned Compose definition. The workflow has no SSH access to production.
Full details are in `DEPLOYMENT.md`.

## Development environments

For plain-language preview/start/stop/cleanup requests, use this repository's
`.agents/skills/dev-app/SKILL.md`. It is project-scoped on both the Mac and VPS.

Development and CI use a worktree-local Python 3.12 environment with uv. Docker
is reserved for production images and deployment. No production credentials are
needed for setup or the tests.

From a clean retained checkout, create one environment for a task:

```bash
make worktree-create NAME=example-task
# cd to the printed worktree path
make setup
make dev-start
make test
```

The helper creates branch `codex/example-task` in
`<main-checkout-name>-worktrees/example-task`. Reuse a task's suitable worktree
across turns. Each server has its own allocated port, log, control socket and
Playwright name recorded in `.dev/session.json`. Report the name, branch, path,
URL and Playwright session when handing it to the user.

The server runs independently of the launching command and remains available
after a turn ends or the phone disconnects. Use `make dev` only for an interactive
foreground terminal. Bind addresses are restricted to loopback or Tailscale;
`DEV_HOST=127.0.0.1 make dev-start` forces local-only access. Python changes need
a stop/start; static/template edits appear on refresh.

Use `make dev-status` for the URL and `make dev-stop` to stop only this worktree.
`make dev-prune` removes stale session metadata after a crash. After stopping,
`make dev-clean` also removes the development log. Use only the recorded named
Playwright session; close it when finished and never use `close-all`.

When the user says they are finished with a named environment, close its named
browser session and, from a retained checkout, run:

```bash
make worktree-check NAME=example-task
make worktree-finish NAME=example-task
```

Finish repeats the checks, stops only that server, removes the worktree and its
disposable logs/virtualenv/caches, and retains the branch. Ending an agent turn
does not authorize cleanup. Never force removal, discard work, delete branches,
or commit/push simply to bypass a guard without the user's authorization.
Report what was removed and the retained branch.

Only helper-created environments are eligible. Cleanup refuses tracked or
untracked changes, unpublished commits, locked/detached/switched worktrees and
unexpected ignored files such as `.env` or model checkpoints. It fetches
`origin` to verify that HEAD is preserved on a remote branch; fetch failure
also refuses cleanup. Do not run competing commands in the same environment.
Other environments remain independent. See `docs/local-development.md`.

Never run development commands against production Compose stacks, containers,
credentials or writable production data.

## Project development notes

The development adapter reads checkpoint files without modifying them. It
checks `DEV_WEIGHTS_DIR` (or `WEIGHTS_DIR`), the worktree/main checkout's
`weights/`, then existing `/srv/gpt/weights` on the VPS. Treat shared model files
as read-only: do not train into or overwrite that directory. Save new checkpoints
inside the task's worktree and preserve them explicitly before cleanup. Native
processes run as the current user, so this is a read-only usage convention rather
than a separate filesystem sandbox.

With no checkpoints, the landing page still works but has no model demos.
Models load lazily; each running process has its own model memory. Prefer small
models and avoid loading several large models on the VPS simultaneously.
Linux setup uses CPU-only PyTorch; Mac setup retains native/MPS support. The
default setup covers the web app and chapter-code tests. Notebook extras in
`requirements.txt` can be installed into the same virtualenv when needed.
