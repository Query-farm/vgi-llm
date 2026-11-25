# vgi-llm — dev and test targets. `make help` lists them. Requires uv.
#
#   make test       # unit tests (pytest) + SQL tests over stdio, launch and http
#   make test-sql   # test/sql/*.test through DuckDB (haybarn-unittest + the signed
#                   # community `vgi` extension) via ci/run-integration.sh
#   make check      # lint + tests
#
# The keyless legs always run (embeddings are local ONNX). live.test is gated on
# ANTHROPIC_API_KEY via `require-env`; without a key it is skipped, and
# ci/run-integration.sh allows exactly that skip.

HAYBARN_VERSION  ?= 2.0.1rc3
WORKER           := $(CURDIR)/.venv/bin/vgi-llm-worker
TOOLS            := $(CURDIR)/.tools
HAYBARN_UNITTEST := $(TOOLS)/bin/haybarn-unittest
TRANSPORTS       ?= stdio launch http

build: ## Install the worker (all extras: HTTP serving needs `http`) into .venv
	uv sync --locked --all-extras

tools: ## Install the pinned SQL test runner into .tools
	@test -x $(HAYBARN_UNITTEST) || { uv venv -q $(TOOLS) && uv pip install -q -p $(TOOLS) haybarn-unittest==$(HAYBARN_VERSION); }

test-sql: build tools warm ## SQL tests through DuckDB over each transport: make test-sql TRANSPORTS=launch
	@set -e; for t in $(TRANSPORTS); do \
	  WORKER=$(WORKER) HAYBARN_UNITTEST=$(HAYBARN_UNITTEST) TRANSPORT=$$t ci/run-integration.sh; \
	done

test-sql-all: test-sql ## Alias: SQL tests over every transport

.DEFAULT_GOAL := help
.PHONY: help build tools warm test test-unit test-sql test-sql-all lint check

help: ## List the targets
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  %-14s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

# So the worker's first ai_embed query does not pay the model download inline
# while the runner is mid-assertion.
warm: build ## Download the local embedding model
	uv run --quiet python -c "from vgi_llm import models; models.warm_up()"

test-unit: build ## Unit tests (pytest)
	uv run pytest -n auto -q

test: test-unit test-sql ## Unit + SQL tests

check: lint test ## Everything CI runs

lint: build ## ruff, mypy, pydoclint
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_llm/
	uvx pydoclint@0.9.1 --config pyproject.toml vgi_llm/
