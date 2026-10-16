IMAGE_NAME = gpt
UV ?= uv
PYTHON = .venv/bin/python
ENV_FILE ?= $(shell if [ -f .env ]; then echo .env; else echo .env.example; fi)
COMPOSE_PROD = ENV_FILE=$(ENV_FILE) docker compose -f compose.yml -f compose.prod.yml
TORCH_BACKEND = $(shell if [ "$$(uname -s)" = Linux ]; then echo --torch-backend=cpu; fi)

# Native development: one environment per worktree.
dev-config:
	python3 scripts/dev_config.py check

worktree-create:
	python3 scripts/dev_worktree.py create "$(NAME)"

worktree-list:
	python3 scripts/dev_worktree.py list

worktree-check:
	python3 scripts/dev_worktree.py check "$(NAME)"

worktree-finish:
	python3 scripts/dev_worktree.py finish "$(NAME)"

setup:
	@if [ ! -x $(PYTHON) ]; then $(UV) venv --python 3.12 .venv; fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Existing .venv is not Python 3.12; recreate it before running make setup")'
	$(UV) pip install --python $(PYTHON) $(TORCH_BACKEND) -r requirements-dev.txt

dev:
	$(PYTHON) scripts/dev_session.py start

dev-start:
	$(PYTHON) scripts/dev_session.py start-detached

dev-status:
	$(PYTHON) scripts/dev_session.py status

dev-stop:
	$(PYTHON) scripts/dev_session.py stop

dev-prune:
	$(PYTHON) scripts/dev_session.py prune

dev-clean:
	$(PYTHON) scripts/dev_session.py clean

test:
	$(PYTHON) -m pytest tests/ -v

# Production images and manual operation.
prod-build:
	docker build -t $(IMAGE_NAME) .

prod-start:
	$(COMPOSE_PROD) up -d --pull never gpt

prod-stop:
	$(COMPOSE_PROD) stop gpt

prod-restart:
	$(COMPOSE_PROD) up -d --pull never gpt

.PHONY: dev-config worktree-create worktree-list worktree-check worktree-finish setup dev dev-start dev-status dev-stop dev-prune dev-clean test prod-build prod-start prod-stop prod-restart
