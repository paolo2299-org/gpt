# Local development

## Plain-language previews

This repository includes `.agents/skills/dev-app/`, a project-scoped Codex skill.
Say **"please run the app"**, **"stop the app"**, or
**"I'm finished testing, clean up this environment"**. The agent chooses a name,
uses this task's worktree, reports the URL, and applies the existing cleanup
checks. No user-level skill installation is needed on the Mac or VPS. New
worktrees inherit it from Git; older worktrees need a commit containing the skill.
The explicit fallback is `$dev-app`. This skill does not apply to other projects.

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

## App-specific setup

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

## Safety and recovery

Install uv once, then run `make setup` in each worktree. `.venv` is never copied
between worktrees. Names use 1–48 lowercase letters, digits and hyphens, starting
with a letter or digit. To choose a base explicitly, use
`python3 scripts/dev_worktree.py create example-task --base origin/main`.
`make worktree-list` lists the registry records under the shared Git directory.

The disposable ignored paths are `.venv/`, Python `__pycache__/`,
`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` and the known `.dev/` log,
lock, session JSON and control socket. Do not put valuable files in cache or
virtualenv directories. Other ignored files, including local `.env`, model
checkpoints, downloaded datasets and browser screenshots, block removal.
Preserve them explicitly with the user before retrying. The helper never
modifies assets outside the worktree or deletes branches.

The publication check uses a fresh fetch of `origin`. A local branch or tag
alone is insufficient. Squash-merged commits may need manual review because
original commits are not ancestors of the squash. A late refusal after shutdown
can leave the worktree's server stopped; use `make dev-start` to resume. If Codex
started the server outside its command sandbox, stopping it may also need the
appropriate command permission. Status can still report the recorded URL using
the worktree lock when its control socket is unavailable.

There is no timer or background deletion. Leave other agents' worktrees and
servers alone. The retained checkout is the controller used to finish an
environment; never finish from the worktree being removed.

Production images and manual operations use `make prod-build`,
`make prod-start`, `make prod-stop` and `make prod-restart`. The authoritative
VPS Compose definitions and deployment timers live in hetzner-vm-setup.

## Private configuration shared by worktrees

Managed native servers automatically read `~/.config/project-dev/gpt.json`
(or `$XDG_CONFIG_HOME/project-dev/gpt.json`). This is private host
configuration, not a file to commit or copy into every worktree. No file is
required for basic previews. `make dev-config` validates it and reports variable
names only; it never prints values. Restart the server after changing config.

Create the directory with mode `700` and the file with mode `600`, owned by the
account running development. Symlinked files/directories, insecure permissions,
invalid JSON and unsupported variables are refused. The format is a JSON object
mapping these supported names to string values:

- `DEV_WEIGHTS_DIR`
- `MODEL_DEVICE`
- `SITE_TITLE`
- `OMP_NUM_THREADS`
- `DEV_HOST`

Use `DEV_CONFIG_FILE=/absolute/path/to/private.json` for an explicit alternative;
an explicit missing file is an error. Use `DEV_CONFIG_DISABLED=1` for an isolated
run without host config. Precedence for managed servers is inherited environment,
then a worktree-local .env, then private host config, then defaults.
The launcher still enforces its development mode, private bind addresses and
existing development safety rules.

Only native development entrypoints load this file; production and ordinary
tests do not. Nothing is exported to other projects or shell startup files.
For an explicitly requested ancillary command (e.g. a scraper), use
`python3 scripts/dev_config.py run -- <command> [arguments...]`. It passes only
this project's allowed configuration to that child process. This wrapper does
not load a worktree-local .env. Do not run scrapers or paid API calls just to
start/verify a preview, print secret values, or copy production environment files.
A user may authorise provisioning a particular existing API key separately.

Cleanup never touches the host config. Existing worktree-local .env files still
block guarded removal; preserve/resolve them explicitly instead of weakening the
guard. Shared configs/credentials are readable by same-user project code; native
worktrees are not a security sandbox for untrusted code.
