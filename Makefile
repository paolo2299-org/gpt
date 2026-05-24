IMAGE_NAME = gpt
ENV_FILE ?= $(shell if [ -f .env ]; then echo .env; else echo .env.example; fi)
COMPOSE = ENV_FILE=$(ENV_FILE) docker compose -f compose.yml -f compose.dev.yml
COMPOSE_PROD = ENV_FILE=$(ENV_FILE) docker compose -f compose.yml -f compose.prod.yml

FLASK ?= $(shell if [ -x .venv/bin/flask ]; then echo .venv/bin/flask; else echo flask; fi)
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PORT ?= 5000

flask-run:
	$(FLASK) --app app run --debug --no-reload --port $(PORT)

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
	$(COMPOSE_PROD) up -d gpt

prod-stop:
	$(COMPOSE_PROD) stop gpt

prod-restart:
	$(COMPOSE_PROD) restart gpt

.PHONY: flask-run dev build run test docker-test down prod-start prod-stop prod-restart
