# gpubox-cli (`gpb`)

Customer-facing CLI for [GPUBox](https://gpubox.ai) — UK-sovereign AI inference.
One binary, every endpoint: chat, embeddings, transcription, fine-tuning,
hosting, vault search, custom assistants, SSO admin.

> Status: **v0.1.0** — early access. Most endpoints follow GPUBox prod;
> some (Vault, Assistants, SSO) require backend Waves 7.x to be merged.
> See [Backend coordination](#backend-coordination) for which commands
> need which backend PR.

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
gpb embed "RAG retrieval target" --json | jq '.data[0].embedding | length'
gpb transcribe ./meeting.mp3
```

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
gpb chat            one-shot or interactive REPL (--interactive)
gpb embed           one-shot embedding
gpb transcribe      Whisper transcription of an audio file

gpb auth login|status|logout
gpb profile list|use|remove
gpb config get|set
gpb signup

gpb billing balance|topup|history
gpb training submit|list|status|watch|download|cancel
gpb hosting list|promote|tier|delete
gpb vault enable|disable|search
gpb vault conversations list|get
gpb vault corpora list|create
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

## Backend coordination

Some commands target endpoints that ship with backend Waves still in PR.
The CLI is wired now so it Just Works the day each PR deploys; until
then, those commands surface the gateway's 404.

| Surface                       | Backend PR | State (as of 2026-05-07) |
| ----------------------------- | ---------- | ------------------------ |
| chat / embed / transcribe     | live       | works on prod            |
| billing balance / topup       | live       | works on prod            |
| training / hosting            | PR #6      | merge → CLI works        |
| vault conversations           | PR #7      | merge → CLI works        |
| guardrails                    | PR #8      | (no CLI yet — server-side)|
| vault rag corpora             | PR #9      | merge → CLI works        |
| assistants                    | PR #10     | merge → CLI works        |
| users / oidc                  | PR #11     | merge → CLI works        |

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

## License

Apache-2.0. See [LICENSE](LICENSE).
