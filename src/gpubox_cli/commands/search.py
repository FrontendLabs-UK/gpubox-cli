"""`gpb search "<query>"` — GPUBox V1.5 W2 unified search + grounded synthesis.

Hits the gateway POST /v1/search route (semantic + keyword over BOTH vault docs
AND chat history, workspace-scoped):

    gpb search "<query>"                FIND — ranked hits (table/json)
    gpb search --synthesize "<query>"   SYNTHESIZE — grounded, server-cited write-up

This is DISTINCT from `gpb vault conversations search` (Postgres keyword FTS over
conversations only) — `gpb search` is semantic + cross-source (docs + chat) and
adds a grounded synthesis mode that FAILS CLOSED (refuses, never free-writes)
when there is nothing to ground on.

Workspace scoping: the active workspace (pinned via `gpb workspace use <id>`) is
sent as the X-GPUBox-Workspace header, so results are scoped to that workspace.
Override per-call with `--workspace <id>`.

Path style: the resolved base_url already ends in /v1, so the path is written
WITHOUT the /v1 prefix ("/search").
"""
from __future__ import annotations

from typing import Optional

import typer

from gpubox_cli import config as cfg
from gpubox_cli.client import ClientConfig, GPUBoxClient, exit_on_error
from gpubox_cli.output import OutputCtx, emit_json, emit_text

# Must match commands/workspace.py — the pinned active-workspace config key.
_ACTIVE_WORKSPACE_KEY = "active_workspace"
WORKSPACE_HEADER = "X-GPUBox-Workspace"


def _build_client(ctx_obj: dict) -> GPUBoxClient:
    resolved = cfg.resolve(
        profile_override=ctx_obj.get("profile"),
        api_key_override=ctx_obj.get("api_key"),
        base_url_override=ctx_obj.get("base_url"),
    )
    return GPUBoxClient(ClientConfig(api_key=resolved.api_key, base_url=resolved.base_url))


def _active_workspace() -> Optional[str]:
    settings = cfg.load_settings()
    val = settings.extra.get(_ACTIVE_WORKSPACE_KEY)
    return str(val) if val else None


def _workspace_headers(override: Optional[str]) -> Optional[dict]:
    ws = override or _active_workspace()
    return {WORKSPACE_HEADER: ws} if ws else None


@exit_on_error
def run(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="The natural-language search query."),
    synthesize: bool = typer.Option(
        False, "--synthesize", "-s",
        help="Return a grounded, server-cited write-up instead of ranked hits "
             "(fails closed: refuses if there is nothing to ground on).",
    ),
    k: int = typer.Option(10, "--k", "-n", help="Hits / evidence cards (default 10)."),
    sources: Optional[str] = typer.Option(
        None, "--sources",
        help="Comma-separated source restriction: docs,chat (default both).",
    ),
    rerank: bool = typer.Option(False, "--rerank", help="Cross-encoder rerank (slower)."),
    min_similarity: float = typer.Option(0.0, "--min-similarity"),
    workspace: Optional[str] = typer.Option(
        None, "--workspace",
        help="Scope to a workspace id (default: the pinned active workspace).",
    ),
) -> None:
    """Unified semantic search over vault docs + chat history.

    FIND (default) prints ranked hits. SYNTHESIZE (--synthesize) prints a
    grounded write-up with [n] citation markers resolved to real doc/chat ids.
    """
    ctx_obj = ctx.obj or {}
    out: OutputCtx = ctx_obj.get("output", OutputCtx())

    body: dict = {
        "query": query,
        "mode": "synthesize" if synthesize else "find",
        "k": k,
        "rerank": rerank,
        "min_similarity": min_similarity,
    }
    if sources:
        body["sources"] = [s.strip() for s in sources.split(",") if s.strip()]

    with _build_client(ctx_obj) as client:
        resp = client.request(
            "POST", "/search", json_body=body,
            extra_headers=_workspace_headers(workspace),
        )

    if out.json_mode:
        emit_json(out, resp)
        return

    if synthesize:
        _render_synthesize(out, resp)
    else:
        _render_find(out, resp)


def _render_find(out: OutputCtx, resp: dict) -> None:
    hits = resp.get("hits", []) if isinstance(resp, dict) else []
    if not hits:
        emit_text(out, "(no matches)")
        return
    for hit in hits:
        snippet = (hit.get("snippet", "") or "").replace("\n", " ")[:80]
        prov = hit.get("provenance", {}) or {}
        label = prov.get("corpus_name") or prov.get("conversation_id") or "?"
        emit_text(
            out,
            f"{hit.get('kind','?'):<8} {str(hit.get('id','?')):<38} "
            f"score={hit.get('score',0):.3f} [{label}] {snippet}",
        )


def _render_synthesize(out: OutputCtx, resp: dict) -> None:
    grounded = resp.get("grounded", False)
    answer = resp.get("answer", "")
    emit_text(out, answer)
    if not grounded:
        emit_text(out, f"\n(refused — not grounded; status={resp.get('verify_status','?')})")
        return
    cites = resp.get("citations", []) or []
    if cites:
        emit_text(out, "\nCitations:")
        for c in cites:
            prov = c.get("provenance", {}) or {}
            label = prov.get("corpus_name") or prov.get("conversation_id") or "?"
            emit_text(out, f"  {c.get('marker','?')} {c.get('kind','?')}:{c.get('id','?')} [{label}]")
