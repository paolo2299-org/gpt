FLASK ?= .venv/bin/flask
PYTHON ?= .venv/bin/python
PORT ?= 5000

run:
	$(FLASK) --app app run --debug --port $(PORT)

test:
	$(PYTHON) -m pytest tests

.PHONY: run test

