#!/usr/bin/env bash
# Refresh (or drift-check) the pinned gateway OpenAPI snapshot used by the
# contract-conformance tests (tests/contract/gateway-openapi.json).
#
# WHAT THE PIN IS
# ---------------
# The committed snapshot is the CLI's PINNED view of the gateway contract. The
# #13 conformance tests catch CLI-side drift AGAINST this pin (a command building
# a bad request). They do NOT, on their own, catch the GATEWAY changing
# underneath the pin — that is what THIS script + the nightly CI job are for.
#
# CANONICAL SOURCE (durable end-state, gateway PR #386)
# -----------------------------------------------------
# The gateway now publishes its OpenAPI spec as a single source of truth at
# `openapi/openapi.json` on master, byte-canonicalised by its own
# scripts/dump_openapi.py as:
#
#     json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
#     (UTF-8 bytes)
#
# This script's PRIMARY source is that published artifact, and on the canonical
# path we write its BYTES VERBATIM so our pin is a byte-exact mirror of the
# gateway's published contract. That makes drift detection a trivial byte
# compare (see `--check` and the nightly workflow) with no false alarms from
# `ensure_ascii` / trailing-newline formatting differences.
#
# Because the gateway repo is PRIVATE, the unauthenticated raw URL 404s. The
# canonical fetch therefore tries, in order:
#   1. the raw URL (works for any future public mirror, or with a PAT cookie),
#   2. the authenticated GitHub Contents API (Accept: application/vnd.github.raw)
#      using GH_TOKEN / GITHUB_TOKEN / `gh auth token`.
#
# FALLBACKS (when the canonical artifact is unreachable)
# -----------------------------------------------------
#   3. a local gpubox-gateway checkout (GATEWAY_REPO) — regenerated and
#      re-serialised with the SAME canonicalisation so the bytes match,
#   4. the live prod spec URL (GATEWAY_URL) — likewise re-serialised.
#
# FAIL-CLOSED
# -----------
# A fetched spec is validated (valid JSON, >50 paths, has components/schemas)
# in a temp file BEFORE it can overwrite the committed pin. A truncated/empty/
# invalid response can never clobber a good pin.
#
# EXIT CODES (so a caller — the nightly CI — can tell drift from a fetch problem):
#   0  in sync (--check) / pin written (refresh)
#   1  DRIFT: the pin no longer matches the canonical contract (--check only)
#   2  fetch/validation failure (no source reachable, or a spec failed sanity)
#      — NOT drift. The nightly must NOT report "contract drifted" on a 2.
#
# Usage:
#   scripts/refresh-contract.sh                 # refresh the pin from canonical
#   scripts/refresh-contract.sh --check         # drift-check only; never write
#   GATEWAY_REPO=/path/to/gpubox-gateway scripts/refresh-contract.sh
#   GATEWAY_URL=https://api.gpubox.ai/openapi.json scripts/refresh-contract.sh
#
set -euo pipefail

# Exit code reserved for "fetch/validation failed" so the caller distinguishes a
# real contract DRIFT (exit 1) from an inability to even obtain a spec (exit 2).
EXIT_FETCH_FAIL=2

MODE="refresh"
if [ "${1:-}" = "--check" ]; then
  MODE="check"
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--check]" >&2
  exit "$EXIT_FETCH_FAIL"
fi

# In --check (the nightly drift guard) we compare ONLY against the gateway's
# CANONICAL published artifact — never the local-repo or live-prod fallbacks.
# Falling back to prod here would let a master-vs-prod skew show a false green:
# master could drift while prod lags, and a prod compare would pass. Refresh
# keeps the fallbacks (they re-serialise to the same canonical bytes anyway).
# Override only for local experimentation: CANONICAL_ONLY=0 scripts/...
CANONICAL_ONLY="${CANONICAL_ONLY:-0}"
if [ "$MODE" = "check" ]; then
  CANONICAL_ONLY=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/tests/contract/gateway-openapi.json"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# --- Canonical (primary) source -------------------------------------------
# Owner/repo/branch/path of the gateway's published canonical spec.
GATEWAY_OWNER_REPO="${GATEWAY_OWNER_REPO:-FrontendLabs-UK/gpubox-gateway}"
GATEWAY_REF="${GATEWAY_REF:-master}"
GATEWAY_SPEC_PATH="${GATEWAY_SPEC_PATH:-openapi/openapi.json}"
CANONICAL_RAW_URL="${CANONICAL_RAW_URL:-https://raw.githubusercontent.com/${GATEWAY_OWNER_REPO}/${GATEWAY_REF}/${GATEWAY_SPEC_PATH}}"

# --- Fallback sources ------------------------------------------------------
# Local checkout (regenerate exactly as the gateway does).
GATEWAY_REPO="${GATEWAY_REPO:-${REPO_ROOT}/../gpubox-gateway}"
GATEWAY_PY="${GATEWAY_REPO}/.venv/bin/python"
# Live prod spec.
GATEWAY_URL="${GATEWAY_URL:-https://api.gpubox.ai/openapi.json}"

# Resolve a GitHub token from the usual env vars, falling back to the gh CLI.
github_token() {
  if [ -n "${GH_TOKEN:-}" ]; then echo "$GH_TOKEN"; return 0; fi
  if [ -n "${GITHUB_TOKEN:-}" ]; then echo "$GITHUB_TOKEN"; return 0; fi
  if command -v gh >/dev/null 2>&1; then gh auth token 2>/dev/null && return 0; fi
  return 1
}

# Re-serialise an arbitrary JSON file to the gateway's exact canonical bytes.
# Used by the regen-from-repo / live-URL fallbacks so their output is BYTE
# IDENTICAL to the published artifact (sort_keys, indent=2, ensure_ascii=False,
# trailing newline, UTF-8). The destination temp path is sys.argv[1] (never
# interpolated into the program), so a quote in TMPDIR cannot inject.
canonicalise() {  # canonicalise <src-json> <dest>
  python3 - "$1" "$2" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
text = json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
open(sys.argv[2], "wb").write(text.encode("utf-8"))
PY
}

# 1+2: canonical published artifact (raw URL, then authenticated Contents API).
# Writes the fetched BYTES VERBATIM so the pin mirrors the artifact exactly.
fetch_canonical() {
  # 1. Unauthenticated raw URL (public mirror / PAT cookie). -f => fail on 4xx/5xx.
  if curl -fsSL "$CANONICAL_RAW_URL" -o "$TMP" 2>/dev/null && [ -s "$TMP" ]; then
    echo ">> fetched canonical spec (raw URL): $CANONICAL_RAW_URL" >&2
    return 0
  fi
  # 2. Authenticated GitHub Contents API (the working path for the PRIVATE repo).
  local token api_url
  if token="$(github_token)" && [ -n "$token" ]; then
    api_url="https://api.github.com/repos/${GATEWAY_OWNER_REPO}/contents/${GATEWAY_SPEC_PATH}?ref=${GATEWAY_REF}"
    if curl -fsSL \
         -H "Authorization: Bearer ${token}" \
         -H "Accept: application/vnd.github.raw" \
         -H "X-GitHub-Api-Version: 2022-11-28" \
         "$api_url" -o "$TMP" 2>/dev/null && [ -s "$TMP" ]; then
      echo ">> fetched canonical spec (authenticated API): $api_url" >&2
      return 0
    fi
  fi
  return 1
}

# 3: local gateway checkout — regenerate, then canonicalise to match.
regen_from_repo() {
  [ -x "$GATEWAY_PY" ] || return 1
  echo ">> regenerating from gateway repo: $GATEWAY_REPO" >&2
  local raw
  raw="$(mktemp)"
  if ( cd "$GATEWAY_REPO" && "$GATEWAY_PY" -c \
        'import json,sys; from app.main import app; json.dump(app.openapi(), open(sys.argv[1],"w"))' \
        "$raw" ); then
    canonicalise "$raw" "$TMP"
    rm -f "$raw"
    return 0
  fi
  rm -f "$raw"
  return 1
}

# 4: live prod URL — fetch, then canonicalise to match.
regen_from_url() {
  echo ">> falling back to live prod: $GATEWAY_URL" >&2
  local raw
  raw="$(mktemp)"
  if curl -fsSL "$GATEWAY_URL" -o "$raw" 2>/dev/null && [ -s "$raw" ]; then
    canonicalise "$raw" "$TMP"
    rm -f "$raw"
    return 0
  fi
  rm -f "$raw"
  return 1
}

got_spec=1
if fetch_canonical; then
  got_spec=0
elif [ "$CANONICAL_ONLY" = "1" ]; then
  echo "ERROR: could not obtain the CANONICAL gateway spec (fallbacks disabled):" >&2
  echo "       1. canonical raw URL:   $CANONICAL_RAW_URL" >&2
  echo "       2. authenticated API:   repos/${GATEWAY_OWNER_REPO}/contents/${GATEWAY_SPEC_PATH}@${GATEWAY_REF}" >&2
  echo "          (needs GH_TOKEN / GITHUB_TOKEN / gh auth — the gateway repo is PRIVATE)" >&2
  echo "       The drift guard compares ONLY against the canonical artifact — it must" >&2
  echo "       never fall back to local/prod (a master-vs-prod skew would false-green)." >&2
  exit "$EXIT_FETCH_FAIL"
elif regen_from_repo; then
  got_spec=0
elif regen_from_url; then
  got_spec=0
fi

if [ "$got_spec" != "0" ]; then
  echo "ERROR: could not obtain the OpenAPI spec from any source:" >&2
  echo "       1. canonical raw URL:   $CANONICAL_RAW_URL" >&2
  echo "       2. authenticated API:   repos/${GATEWAY_OWNER_REPO}/contents/${GATEWAY_SPEC_PATH}@${GATEWAY_REF}" >&2
  echo "          (needs GH_TOKEN / GITHUB_TOKEN / gh auth — the gateway repo is PRIVATE)" >&2
  echo "       3. local repo gen:      $GATEWAY_PY (set GATEWAY_REPO)" >&2
  echo "       4. live prod URL:       $GATEWAY_URL (set GATEWAY_URL)" >&2
  exit "$EXIT_FETCH_FAIL"
fi

# Fail-closed sanity: must be a plausible OpenAPI document — valid JSON, a
# non-trivial paths object, and a components.schemas object — before we trust it.
# A truncated curl must never get anywhere near the committed pin. Explicit raises
# (not assert) so PYTHONOPTIMIZE / `python -O` cannot disable the guard. Exit 2
# (EXIT_FETCH_FAIL) on any failure so the caller does not mistake it for drift.
if ! python3 - "$TMP" <<'PY'
import json, sys
try:
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
except (ValueError, OSError) as exc:
    raise SystemExit(f"refused: not valid JSON: {exc}")
if not isinstance(spec, dict) or "openapi" not in spec:
    raise SystemExit("refused: missing top-level 'openapi' — not an OpenAPI doc")
paths = spec.get("paths")
if not (isinstance(paths, dict) and len(paths) > 50):
    n = len(paths) if isinstance(paths, dict) else "no"
    raise SystemExit(f"refused: spec has {n} paths (expected >50) — looks truncated")
components = spec.get("components")
schemas = components.get("schemas") if isinstance(components, dict) else None
if not isinstance(schemas, dict) or len(schemas) < 20:
    raise SystemExit("refused: components.schemas missing or implausibly small")
print(f"ok: {len(paths)} paths, {len(schemas)} schemas", file=sys.stderr)
PY
then
  echo "ERROR: fetched spec failed sanity validation — refusing to use it." >&2
  exit "$EXIT_FETCH_FAIL"
fi

if [ "$MODE" = "check" ]; then
  # Drift guard: compare the freshly-fetched canonical spec to the committed pin
  # WITHOUT writing. Byte-compare is exact because the canonical path writes the
  # artifact's bytes verbatim and the fallbacks re-serialise identically.
  if [ ! -f "$DEST" ]; then
    # A missing pin is a repo-state problem, not a contract DRIFT — exit 2 so the
    # nightly does not file a "contract drifted" issue for it.
    echo "ERROR: committed pin $DEST is missing" >&2
    exit "$EXIT_FETCH_FAIL"
  fi
  if cmp -s "$TMP" "$DEST"; then
    echo "ok: pinned snapshot matches the published gateway contract" >&2
    exit 0
  fi
  echo "DRIFT: the pinned snapshot no longer matches the published gateway contract." >&2
  echo "       The GATEWAY contract moved underneath the CLI pin. To resolve:" >&2
  echo "         scripts/refresh-contract.sh   # re-pin to the canonical spec" >&2
  echo "         pytest tests/test_contract_conformance.py tests/test_contract_guard_proof.py" >&2
  echo "       A green run after re-pin = CLI still conforms; a red run = a CLI command must be updated." >&2
  echo "--- diff (canonical vs pinned) ---" >&2
  diff "$DEST" "$TMP" | head -60 >&2 || true
  exit 1
fi

# install (not mv) so the committed pin keeps a sane 0644 mode rather than
# inheriting mktemp's restrictive 0600 (Codex review).
install -m 0644 "$TMP" "$DEST"
rm -f "$TMP"
trap - EXIT
echo ">> wrote $DEST"
echo ">> now run: pytest tests/test_contract_conformance.py tests/test_contract_guard_proof.py"
echo "   a green run = CLI conforms to the refreshed pin."
echo "   a red run after a refresh = the gateway contract moved; update the CLI command(s)."
