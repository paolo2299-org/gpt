IMAGE_NAME = gpt
COMPOSE = docker compose -f compose.yml -f compose.dev.yml
COMPOSE_PROD = docker compose -f compose.yml -f compose.prod.yml

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PORT ?= 5000

# Use `python -m flask` rather than the `flask` console script: console-script
# shebangs hardcode an absolute interpreter path, which breaks if the venv was
# copied/renamed from another project. `python -m` resolves the interpreter itself.
flask-run:
	$(PYTHON) -m flask --app app run --debug --no-reload --port $(PORT)

dev:
	$(COMPOSE) up --build gpt

build:
	docker build -t $(IMAGE_NAME) .

run:
	$(COMPOSE) up --build gpt

test:
	$(PYTHON) -m pytest tests

docker-test:
	$(COMPOSE) run --rm test

down:
	$(COMPOSE) down --remove-orphans

prod-start:
	$(COMPOSE_PROD) up -d --pull never gpt

prod-stop:
	$(COMPOSE_PROD) stop gpt

prod-restart:
	$(COMPOSE_PROD) up -d --pull never gpt

.PHONY: flask-run dev build run test docker-test down prod-start prod-stop prod-restart
