#!/usr/bin/env bash
# Run the SQLLogic (haybarn) suite in test/sql/ against the Python worker, using the
# haybarn DuckDB distribution's unittest runner (which loads the `vgi` extension from
# the community repository).
#
# The bind-only DESCRIBE asserts and the keyless legs (prompt / ai_count_tokens /
# ai_embed / ai_similarity) always run -- embeddings are local ONNX (fastembed), no key.
# The live legs in live.test are gated on ANTHROPIC_API_KEY via `require-env`, so they
# are SKIPPED (not failed) when no key is present -- CI stays green anywhere.
#
# Prerequisites (one-time):
#   uv tool install haybarn-unittest                      # the DuckDB unittest binary
#   echo "INSTALL vgi FROM community;" | uvx haybarn-cli  # install the vgi extension
#   uv sync --extra dev                                   # the worker's Python environment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UNITTEST="${VGI_UNITTEST:-$(command -v haybarn-unittest || true)}"
if [[ -z "$UNITTEST" || ! -x "$UNITTEST" ]]; then
    echo "ERROR: haybarn-unittest not found. Install it with:" >&2
    echo "       uv tool install haybarn-unittest" >&2
    exit 1
fi

# Ensure the vgi community extension is installed for this haybarn version.
if ! echo "LOAD vgi;" | uvx haybarn-cli >/dev/null 2>&1; then
    echo "==> Installing vgi extension from community repository"
    echo "INSTALL vgi FROM community;" | uvx haybarn-cli
fi

# Warm the local embedding model so the worker's first ai_embed query does not pay the
# model download inline while the runner is mid-assertion.
uv run --quiet python -c "from vgi_aisql import models; models.warm_up()" || true

# NOTE: the last arg is a Catch2 test-name filter, not a shell glob. Catch2 only honors a
# trailing `*` wildcard, so use `test/sql/*` (not `test/sql/*.test`).
WORKER="$REPO_ROOT/bin/vgi-aisql-worker"
TEST_GLOB="${1:-test/sql/*}"

echo "==> Running SQLLogic tests"
echo "    worker:   $WORKER"
echo "    unittest: $UNITTEST"
echo "    tests:    $TEST_GLOB"

VGI_TEST_WORKER="$WORKER" \
VGI_WORKER_CATALOG_NAME="aisql" \
    "$UNITTEST" --test-dir "$REPO_ROOT" "$TEST_GLOB"
