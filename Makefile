.PHONY: help install dev lint format test coverage build check serve verify clean

PY ?= python3
VENV ?= .venv

help:
	@echo "make install   install runtime package (serve extras)"
	@echo "make dev       install package + dev extras into $(VENV)"
	@echo "make lint      ruff check + mypy"
	@echo "make format    ruff format"
	@echo "make test      pytest"
	@echo "make coverage  pytest with coverage report"
	@echo "make build     build sdist + wheel into dist/"
	@echo "make check     lint + test + build (the CI gate)"
	@echo "make serve     run the HTTP service on :8100"
	@echo "make verify    run the acceptance check against :8100"
	@echo "make clean     remove build artefacts and caches"

install:
	$(PY) -m pip install -e ".[serve]"

dev:
	test -d $(VENV) || $(PY) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -U pip
	$(VENV)/bin/python -m pip install -e ".[serve,dev]"

lint:
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/mypy

format:
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

test:
	$(VENV)/bin/pytest

coverage:
	$(VENV)/bin/pytest --cov=sichuan_speech --cov-report=term-missing

build:
	$(VENV)/bin/python -m build

check: lint test build

serve:
	$(VENV)/bin/sichuan-speech serve

verify:
	$(VENV)/bin/sichuan-speech verify

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
