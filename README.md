# gpubox-cli (`gpb`)

Customer-facing CLI for [GPUBox](https://gpubox.ai) — UK-sovereign AI inference.
One binary, every endpoint: chat, embeddings, transcription, fine-tuning,
hosting, vault search, custom assistants, SSO admin.

> Status: **v0.1.1** (early access). Every subcommand maps to a live
> gateway endpoint on prod. See [Endpoint mapping](#endpoint-mapping) for
> the full `gpb <command>` to `<METHOD> /v1/<path>` table.

---

## Install

```bash
# Recommended — isolated install via pipx (no venv pollution)
pipx install gpubox-cli

# Or vanilla pip
pip install gpubox-cli

# Both `gpb` and `gpubox` are wired as entry points
gpb --help
```

Once `pip install gpubox-cli` is published. Until then:

```bash
git clone https://github.com/FrontendLabs-UK/gpubox-cli && cd gpubox-cli
pip install -e .
```

## Quickstart

```bash
gpb signup --email you@example.com    # opens https://gpubox.ai/signup
gpb auth login                         # paste your gpb_live_… key (input hidden)
gpb auth status                        # confirm identity
gpb chat "what's the capital of Nigeria?"
gpb chat "what's wrong in this screenshot?" --image bug.png   # vision, auto-routed
gpb embed "RAG retrieval target" --json | jq '.data[0].embedding | length'
gpb transcribe ./meeting.mp3
```

## Vision (image input)

Attach images with `--image` (alias `-I`, repeatable). The CLI builds
OpenAI-compatible multimodal content and, unless you pin `--model`, routes
the turn to the vision model `qwen2.5-vl-7b-instruct`:

```bash
# Local file (read + sent inline as a base64 data-URI)
gpb chat "describe this UI and any errors you see" -I screenshot.png

# Multiple images in one prompt
gpb chat "what changed between these?" -I before.png -I after.png

# An https URL or a data: URI also works
gpb chat "read the chart" -I https://example.com/chart.png

# Image-only — defaults the text to "Describe this image."
gpb chat -I receipt.jpg
```

In the REPL, `/image <path-or-url>` attaches an image to your next message
(that turn auto-routes to the vision model). Good for screenshot/UI
analysis, OCR, chart and document-image reading, and visual Q&A.

For scripts and CI, prefer environment variables:

```bash
export GPUBOX_API_KEY=gpb_live_...
export GPUBOX_API_URL=https://api.gpubox.ai/v1   # default
gpb --json chat "summarise: $(cat ticket.txt)" | jq -r '.choices[0].message.content'
```

## Profiles

If you work across multiple tenants (consultant juggling 3 clients, or
swapping prod ↔ staging), profiles keep keys separate:

```bash
gpb auth login --profile acme           # save key into "acme"
gpb auth login --profile prod-internal  # different key
gpb profile list
gpb profile use acme                    # default for the next command

# Or one-shot override:
gpb --profile prod-internal chat "ping"
```

Profiles + settings live in:

| OS      | Path                                              |
| ------- | ------------------------------------------------- |
| Linux   | `~/.config/gpubox/`                               |
| macOS   | `~/Library/Application Support/gpubox/`           |
| Windows | `%APPDATA%\gpubox\`                               |

Override via `GPB_CONFIG_DIR` (useful in CI / containers without `$HOME`).
The credentials file is mode `0600` on POSIX.

## Output controls

Every command honours these global flags:

| Flag           | Behaviour                                   |
| -------------- | ------------------------------------------- |
| `--json`, `-j` | Machine-readable JSON to stdout             |
| `--quiet`, `-q`| Suppress info output                        |
| `--verbose`,`-v`| Debug/trace                                |
| `--no-color`   | Strip colour (also via `NO_COLOR=1`)        |
| `--profile`,`-p`| Pick a named profile                       |
| `--api-key`    | One-shot key override                       |
| `--base-url`   | One-shot endpoint override                  |

Streaming chat tokens render live in a TTY. When stdout is piped or
`--json` is set, the CLI buffers the full response — so
`gpb chat "..." | tee out.txt` always gets the whole answer, never half.

## Commands

```
gpb chat            one-shot or interactive REPL (--interactive); --image for vision
gpb embed           one-shot embedding
gpb transcribe      Whisper transcription of an audio file

gpb auth login|status|logout
gpb profile list|use|remove
gpb config get|set
gpb signup

gpb billing balance|topup|history
gpb training submit|list|status|watch|download|cancel
gpb hosting list|promote|tier|delete
gpb vault enable|disable|search          # enable/disable are operator-only (no public route)
gpb vault conversations list|get|delete
gpb vault corpora list|get|create|delete
gpb assistants list|create|update|run|delete
gpb users invite|list
gpb users oidc create|list
```

Run any command with `--help` for full options.

## Top-up flow (UK + NG)

```bash
gpb billing topup --amount-gbp 10           # opens Stripe checkout
gpb billing topup --amount-ngn 10000        # opens Paystack checkout
```

If the wallet hits zero mid-stream, the CLI fails fast with the topup
URL on stderr — no half-streamed JSON blob to clean up.

## Telemetry

**Zero. We do not phone home.** No anonymous metrics, no version pings,
no opt-out toggle for telemetry that doesn't exist. If something breaks,
run `gpb auth status -j` and paste the output into a GitHub issue.

## Endpoint mapping

Every `gpb` subcommand below maps to a single gateway route. Paths are
relative to the API base (default `https://api.gpubox.ai/v1`, override
with `GPUBOX_API_URL` or `--base-url`). The gateway's full OpenAPI spec
lives at `https://api.gpubox.ai/docs`.

| CLI command                          | HTTP                          | Notes                                                |
| ------------------------------------ | ----------------------------- | ---------------------------------------------------- |
| `gpb chat`                           | POST `/v1/chat/completions`   | SSE stream in a TTY, buffered otherwise; `--image` sends multimodal content to the vision model (`qwen2.5-vl-7b-instruct`) |
| `gpb embed`                          | POST `/v1/embeddings`         | default model `BAAI/bge-m3`                          |
| `gpb transcribe`                     | POST `/v1/audio/transcriptions` | multipart upload; Whisper-compatible                 |
| `gpb signup`                         | (browser)                     | opens `https://gpubox.ai/signup` (no API call)       |
| `gpb auth login`                     | (local)                       | writes credentials file (mode 0600)                  |
| `gpb auth status`                    | GET `/v1/auth/whoami`         | falls back to "unverified" on 404/405                |
| `gpb auth logout`                    | (local)                       | clears API key, keeps base_url + default_model       |
| `gpb profile list/use/remove`        | (local)                       | edits credentials/config files only                  |
| `gpb config get/set`                 | (local)                       | edits `config.toml` only                             |
| `gpb billing balance`                | GET `/v1/billing/balance`     |                                                      |
| `gpb billing history`                | GET `/v1/billing/balance`     | reads `recent_topups` from balance                   |
| `gpb billing topup --amount-gbp`     | POST `/v1/billing/checkout-sessions` | Stripe checkout (pence)                       |
| `gpb billing topup --amount-ngn`     | POST `/v1/billing/paystack/initialize` | Paystack checkout (kobo)                    |
| `gpb training submit`                | POST `/v1/training/runs`      | idempotent; vault corpus (`--since`/`--until`); tunables via `hyperparams` |
| `gpb training list`                  | GET `/v1/training/runs`       | `--status` filter, `--limit`                         |
| `gpb training status <run_id>`       | GET `/v1/training/runs/{run_id}` |                                                  |
| `gpb training watch <run_id>`        | GET `/v1/training/runs/{run_id}` | polls every `--interval` seconds                  |
| `gpb training download <run_id>`     | GET `/v1/training/runs/{run_id}/download` | JSON `{url,…}`; CLI fetches the signed URL → disk, verifies sha256 |
| `gpb training cancel <run_id>`       | POST `/v1/training/runs/{run_id}/cancel` |                                            |
| `gpb hosting list`                   | GET `/v1/hosting/models`      |                                                      |
| `gpb hosting promote <run_id>`       | POST `/v1/hosting/models`     | idempotent; required `--name`; `--tier cold/warm/always_hot` |
| `gpb hosting tier <model_id>`        | POST `/v1/hosting/models/{model_id}/transition` | body `{hosting_tier}`                  |
| `gpb hosting delete <model_id>`      | DELETE `/v1/hosting/models/{model_id}` |                                             |
| `gpb vault enable`                   | (none — operator-only)        | no public route; prints "email support@gpubox.ai"    |
| `gpb vault disable`                  | (none — operator-only)        | no public route; prints "email support@gpubox.ai"    |
| `gpb vault search`                   | POST `/v1/conversations/search` | keyword FTS (`--mode fts`/`substring`), not semantic |
| `gpb vault conversations list`       | GET `/v1/conversations`       |                                                      |
| `gpb vault conversations get <id>`   | GET `/v1/conversations/{id}` (+ `/messages`) | metadata + message history             |
| `gpb vault conversations delete <id>`| DELETE `/v1/conversations/{id}` | soft-delete                                        |
| `gpb vault corpora list`             | GET `/v1/corpora`             |                                                      |
| `gpb vault corpora get <id>`         | GET `/v1/corpora/{id}`        |                                                      |
| `gpb vault corpora delete <id>`      | DELETE `/v1/corpora/{id}`     | soft-delete                                          |
| `gpb vault corpora create`           | POST `/v1/corpora`            | `{name, source_type, content}`; `--from-file` PDF → multipart `/v1/corpora/upload` |
| `gpb assistants list`                | GET `/v1/assistants`          |                                                      |
| `gpb assistants create`              | POST `/v1/assistants`         | idempotent; required `--slug` + `--name`             |
| `gpb assistants update <id>`         | POST `/v1/assistants/{id}`    | creates a new version                                |
| `gpb assistants run <id>`            | POST `/v1/chat/completions`   | model alias `asst_<id>` (single user turn)           |
| `gpb assistants delete <id>`         | DELETE `/v1/assistants/{id}`  |                                                      |
| `gpb users invite <email>`           | POST `/v1/tenants/{tenant_id}/users` | tenant from `--tenant` or `GPUBOX_TENANT_ID` |
| `gpb users list`                     | GET `/v1/tenants/{tenant_id}/users` |                                               |
| `gpb users oidc create`              | POST `/v1/oidc/clients`       | `redirect_uris` is a list                            |
| `gpb users oidc list`                | GET `/v1/oidc/clients`        |                                                      |

If a row doesn't match prod (404, schema drift), the gateway's
OpenAPI is the source of truth. Open an issue with the diff.

## Design rationale (cdxk round-table)

This v0.1 was locked through a Claude × Codex × Kimi round-table.
Highlights:

- **Python + typer + httpx + rich** — matches the OpenAI-SDK install
  path our customers already know. Sync httpx; CLI doesn't benefit
  from async.
- **PyPI primary, Homebrew tap roadmap** — `pipx install` recommended
  to dodge venv pollution. NPM/Rust binaries deferred to v0.2 if NG
  bandwidth becomes a real blocker (Kimi's contrarian point — solid
  but not day-one).
- **`gpb` short name + `gpubox` alias** — `pip install gpubox-cli`
  exposes both entry points; consistent with `gh`/`kubectl`/`gcloud`.
- **Paste-API-key v0.1, OIDC device flow v0.2** — paste-key matches
  the dashboard mental model. OIDC arrives when backend Wave 7.5 ships.
- **File-based 0600 credentials, no system keyring** — keyring on
  headless boxes is painful. v0.2 may add it as opt-in.
- **TTY-aware streaming, JSON-on-flag only** — no surprise JSON when
  piping plain text. Diagnostics on stderr; stdout is clean for pipes.
- **Conversation persistence is opt-in** — `--save-session <path>`. We
  never silently dump prompts to disk (privacy lock for sovereign-UK
  positioning).
- **Zero telemetry. No PyPI version ping.** Trust signal alignment.
- **Profiles are first-class from day 1** — `gpb profile list/use/remove`.
- **Public Apache-2.0** — open-source CLI is the right trust signal
  for an inference SaaS.

Codex review threads: see PR description for thread IDs.

## Development

```bash
git clone https://github.com/FrontendLabs-UK/gpubox-cli
cd gpubox-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
```

### Contract conformance (CLI ↔ gateway request shapes)

Most of the gateway's mutating request models are Pydantic `extra='forbid'` (a
few, e.g. `WorkspaceCreate`, are not): on a `forbid` model an unknown key 422s, a
missing required field 422s on any model, and a wrong method/path 404/405s.
Several `gpb` commands once shipped requests the live gateway rejected. The
contract harness is the systematic guard against that whole class of drift, and
it follows the snapshot's own `additionalProperties` flag per-model rather than
assuming every model forbids extras.

- `tests/contract/gateway-openapi.json` — the **pinned** gateway OpenAPI snapshot
  (the CLI's authoritative view of the contract).
- `tests/contract/conformance.py` — `assert_request_conforms(...)`: a
  dependency-free validator that resolves the operation for `(method, path)`,
  fails on an **unregistered method/path**, on an **undeclared key** where the
  schema sets `additionalProperties: false`, and on a **missing required**
  property.
- `tests/test_contract_conformance.py` — drives every mutating command through a
  mocked transport and asserts the request it actually builds conforms.
- `tests/test_contract_guard_proof.py` — feeds the eight known-bad historical
  bodies to the validator and asserts each is rejected (proving the guard works).

Refresh the pin when the gateway contract changes:

```bash
# primary: regenerate from a local gpubox-gateway checkout
GATEWAY_REPO=/path/to/gpubox-gateway scripts/refresh-contract.sh
# fallback: pull the live prod spec
GATEWAY_URL=https://api.gpubox.ai/openapi.json scripts/refresh-contract.sh
```

A green run after a refresh means the CLI conforms to the new contract; a **red**
run after a refresh means the gateway moved and a CLI command must be updated.

**Limitations.** The pin can lag the live gateway; the harness validates
*top-level* request *shape*, not *semantics* — it won't catch a wrong-but-valid
value, and it does not yet recurse into nested object/array properties (e.g. a
malformed `messages[]` element), nor validate query params. The eight drifts it
exists to catch were all top-level key/method/path issues. The durable end-state
is the gateway publishing `openapi.json` as a CI artifact the CLI pulls
automatically, instead of a manually-refreshed pin.

## License

Apache-2.0. See [LICENSE](LICENSE).
