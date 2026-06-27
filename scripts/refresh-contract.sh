#!/usr/bin/env bash
# Refresh the pinned gateway OpenAPI snapshot used by the contract-conformance
# tests (tests/contract/gateway-openapi.json).
#
# THE CANONICAL SOURCE is the gateway's PUBLISHED spec — the gateway is the
# single source of truth for its OpenAPI contract. As of gpubox-gateway's
# openapi-publish work, the gateway commits a canonical openapi/openapi.json on
# every contract change (CI fails the gateway PR if the committed spec drifts
# from the code), fetchable at a stable raw URL:
#
#   https://raw.githubusercontent.com/FrontendLabs-UK/gpubox-gateway/master/openapi/openapi.json
#
# This script PULLS that canonical artifact and re-pins it here. It supersedes
# the hand-copied snapshot that gpubox-cli PR #13's "Honest limitations" flagged
# as the staleness gap: PR #13 caught CLI-side drift against the pin, but not the
# gateway moving underneath it. Pulling the canonical artifact closes that gap —
# the pin now tracks the gateway's own published bytes, not a manual copy.
#
#   * Snapshot unchanged after refresh  -> the gateway contract is stable; any
#     conformance failure is genuine CLI drift (a command building a bad request).
#   * Snapshot changed + a test now fails -> the GATEWAY contract moved; a CLI
#     command must be updated to match (and the snapshot bump committed with it).
#
# Source precedence (first that succeeds wins):
#   1. CANONICAL_URL  — the gateway's committed openapi/openapi.json raw URL.
#                       PREFERRED: preserves the gateway's EXACT published bytes,
#                       so the pin is byte-identical to the canonical artifact
#                       (the gateway's own --check gate guarantees those bytes
#                       match its code). Needs only network, no gateway checkout.
#   2. GATEWAY_REPO   — a local gpubox-gateway checkout's dump_openapi.py (or the
#                       app venv directly). Offline/exact; matches the canonical
#                       byte format (sort_keys + ensure_ascii=False).
#   3. GATEWAY_URL    — live prod /openapi.json. Last resort (a deploy lag means
#                       prod can trail master); re-serialised to the canonical
#                       byte format so the pin format stays stable.
#
# The nightly CI drift monitor (.github/workflows/contract-drift.yml) runs this
# with REQUIRE_CANONICAL=1 (canonical URL ONLY — no silent prod fallback) and
# FAILS if the pin no longer matches, so gateway-side drift is caught
# automatically, not only when a human remembers to refresh.
#
# Usage:
#   scripts/refresh-contract.sh                          # auto: canonical URL, then local repo, then prod
#   REQUIRE_CANONICAL=1 scripts/refresh-contract.sh      # strict: canonical URL only, hard-fail otherwise (the monitor)
#   CANONICAL_URL=<raw-url> scripts/refresh-contract.sh  # override the canonical source
#   GATEWAY_REPO=/path/to/gpubox-gateway scripts/refresh-contract.sh
#   GATEWAY_URL=https://api.gpubox.ai/openapi.json scripts/refresh-contract.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/tests/contract/gateway-openapi.json"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# Canonical artifact — the gateway's committed openapi/openapi.json on master.
CANONICAL_URL="${CANONICAL_URL:-https://raw.githubusercontent.com/FrontendLabs-UK/gpubox-gateway/master/openapi/openapi.json}"
# Local gateway checkout (offline/exact fallback). Override with GATEWAY_REPO.
GATEWAY_REPO="${GATEWAY_REPO:-${REPO_ROOT}/../gpubox-gateway}"
GATEWAY_PY="${GATEWAY_REPO}/.venv/bin/python"
# Live prod (last resort — may trail master between deploys).
GATEWAY_URL="${GATEWAY_URL:-https://api.gpubox.ai/openapi.json}"
# Strict mode: when set (the nightly drift monitor sets it), ONLY the canonical
# URL is acceptable — no local-repo / prod fallback. Without this, a 404 /
# rename / outage on the canonical URL would let the monitor silently validate
# the pin against PROD (which can trail master), so it could pass while never
# checking the canonical published spec — defeating its purpose (Codex HIGH).
# Manual/local refreshes leave it unset and keep the full fallback chain.
REQUIRE_CANONICAL="${REQUIRE_CANONICAL:-}"

# The destination temp path is passed as sys.argv[1], NOT interpolated into the
# Python source, so a TMPDIR containing a quote can neither break nor inject the
# `python -c` program. All three paths land the spec in the SAME canonical byte
# format the gateway publishes (indent=2, sort_keys=True, ensure_ascii=False,
# trailing newline) so the pin stays byte-identical to the canonical artifact.

# 1. Canonical raw URL — preserve the gateway's EXACT published bytes verbatim
#    (do NOT re-serialise: the gateway's --check gate already guarantees these
#    bytes match its code, and re-encoding could perturb formatting). We only
#    validate it parses as JSON below.
fetch_from_canonical() {
  echo ">> fetching canonical published spec: $CANONICAL_URL" >&2
  # -f: fail on HTTP errors (a 404 must NOT clobber the pin with an error page).
  curl -fsS "$CANONICAL_URL" -o "$TMP"
}

# 2. Local gateway checkout — prefer dump_openapi.py (the canonical generator);
#    fall back to a direct app.openapi() call matching the canonical format.
regen_from_repo() {
  [ -x "$GATEWAY_PY" ] || return 1
  echo ">> regenerating from local gateway repo: $GATEWAY_REPO" >&2
  if [ -f "$GATEWAY_REPO/scripts/dump_openapi.py" ]; then
    ( cd "$GATEWAY_REPO" && "$GATEWAY_PY" scripts/dump_openapi.py --stdout ) >"$TMP"
  else
    ( cd "$GATEWAY_REPO" && "$GATEWAY_PY" -c \
        'import json,sys; from app.main import app; sys.stdout.write(json.dumps(app.openapi(), indent=2, sort_keys=True, ensure_ascii=False)+"\n")' \
        >"$TMP" )
  fi
}

# 3. Live prod — re-serialise to the canonical byte format (prod serves compact
#    JSON; normalise so the pin format matches the canonical artifact).
regen_from_url() {
  echo ">> falling back to live prod: $GATEWAY_URL" >&2
  curl -fsS "$GATEWAY_URL" \
    | python3 -c 'import json,sys; sys.stdout.write(json.dumps(json.load(sys.stdin), indent=2, sort_keys=True, ensure_ascii=False)+"\n")' \
      >"$TMP"
}

if [ -n "$REQUIRE_CANONICAL" ]; then
  # Strict: the canonical URL is the ONLY acceptable source. A failure here must
  # NOT silently degrade to prod — the monitor would then pass against the wrong
  # spec. Hard-fail so a broken/renamed canonical URL is itself an alert.
  if ! fetch_from_canonical; then
    echo "ERROR: REQUIRE_CANONICAL set but the canonical spec is unreachable:" >&2
    echo "       $CANONICAL_URL" >&2
    echo "       Refusing to fall back to a local repo or prod in strict mode." >&2
    echo "       (Has the gateway repo/path moved, or is the canonical artifact" >&2
    echo "        not yet published? Fix the URL or unset REQUIRE_CANONICAL.)" >&2
    exit 1
  fi
elif fetch_from_canonical; then
  :
elif regen_from_repo; then
  :
elif regen_from_url; then
  :
else
  echo "ERROR: could not fetch the canonical spec ($CANONICAL_URL)," >&2
  echo "       regenerate from a local gateway repo ($GATEWAY_PY missing)," >&2
  echo "       nor fall back to live prod ($GATEWAY_URL)." >&2
  echo "       Set CANONICAL_URL, GATEWAY_REPO, or GATEWAY_URL." >&2
  exit 1
fi

# Sanity: must be valid JSON with a non-trivial paths object before we overwrite
# the committed pin (a truncated curl must never clobber the snapshot).
python3 - "$TMP" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    spec = json.load(fh)
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
