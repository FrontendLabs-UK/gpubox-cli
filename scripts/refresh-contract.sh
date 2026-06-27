#!/usr/bin/env bash
# Refresh the pinned gateway OpenAPI snapshot used by the contract-conformance
# tests (tests/contract/gateway-openapi.json).
#
# The committed snapshot is the CLI's PINNED view of the gateway contract. The
# conformance tests catch CLI-side drift AGAINST this pin. They do NOT, on their
# own, catch the gateway changing underneath the pin — that is what THIS script
# is for: re-pin, then re-run the tests.
#
#   * Snapshot unchanged after refresh  -> the gateway contract is stable; any
#     conformance failure is genuine CLI drift (a command building a bad request).
#   * Snapshot changed + a test now fails -> the GATEWAY contract moved; a CLI
#     command must be updated to match (and the snapshot bump committed with it).
#
# Refresh whenever the gateway contract changes (new endpoint, renamed field,
# tightened required set). See the durable end-state note in the conformance PR:
# the better long-term answer is the gateway publishing openapi.json as a CI
# artifact the CLI pulls, rather than this manual pin.
#
# Usage:
#   scripts/refresh-contract.sh                 # auto: try local gateway, else prod
#   GATEWAY_REPO=/path/to/gpubox-gateway scripts/refresh-contract.sh
#   GATEWAY_URL=https://api.gpubox.ai/openapi.json scripts/refresh-contract.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/tests/contract/gateway-openapi.json"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# Where to find the gateway repo for the primary (offline, exact) path. Override
# with GATEWAY_REPO. The default mirrors the common local checkout layout.
GATEWAY_REPO="${GATEWAY_REPO:-${REPO_ROOT}/../gpubox-gateway}"
GATEWAY_PY="${GATEWAY_REPO}/.venv/bin/python"
GATEWAY_URL="${GATEWAY_URL:-https://api.gpubox.ai/openapi.json}"

# The destination temp path is passed as sys.argv[1], NOT interpolated into the
# Python source, so a TMPDIR containing a quote can neither break nor inject the
# `python -c` program.
regen_from_repo() {
  [ -x "$GATEWAY_PY" ] || return 1
  echo ">> regenerating from gateway repo: $GATEWAY_REPO" >&2
  ( cd "$GATEWAY_REPO" && "$GATEWAY_PY" -c \
      'import json,sys; from app.main import app; json.dump(app.openapi(), open(sys.argv[1],"w"), indent=2, sort_keys=True)' \
      "$TMP" )
}

regen_from_url() {
  echo ">> falling back to live prod: $GATEWAY_URL" >&2
  # -f: fail on HTTP errors; pipe through python to pretty-print deterministically.
  curl -fsS "$GATEWAY_URL" \
    | python3 -c 'import json,sys; json.dump(json.load(sys.stdin), open(sys.argv[1],"w"), indent=2, sort_keys=True)' \
      "$TMP"
}

if regen_from_repo; then
  :
elif regen_from_url; then
  :
else
  echo "ERROR: could not regenerate the OpenAPI snapshot from the gateway repo" >&2
  echo "       ($GATEWAY_PY missing) nor from $GATEWAY_URL." >&2
  echo "       Set GATEWAY_REPO to your gpubox-gateway checkout, or GATEWAY_URL." >&2
  exit 1
fi

# Sanity: must be valid JSON with a non-trivial paths object before we overwrite
# the committed pin (a truncated curl must never clobber the snapshot).
python3 - "$TMP" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
paths = spec.get("paths", {})
# Explicit raises (not assert) so PYTHONOPTIMIZE / `python -O` cannot disable the
# guard and let a truncated spec clobber the committed pin.
if not (isinstance(paths, dict) and len(paths) > 50):
    raise SystemExit(
        f"refused: spec has only {len(paths)} paths (expected >50) — looks truncated"
    )
if "components" not in spec or "schemas" not in spec["components"]:
    raise SystemExit("refused: spec missing components/schemas")
print(f"ok: {len(paths)} paths, {len(spec['components']['schemas'])} schemas")
PY

mv "$TMP" "$DEST"
trap - EXIT
echo ">> wrote $DEST"
echo ">> now run: pytest tests/test_contract_conformance.py tests/test_contract_guard_proof.py"
echo "   a green run = CLI conforms to the refreshed pin."
echo "   a red run after a refresh = the gateway contract moved; update the CLI command(s)."
