"""The eight historical request-shape drifts PR #12 fixed — as fixtures.

Each entry is a request the CLI USED to send and the live gateway REJECTED
(422/404/405). They are kept here, clearly labelled "historical drift — must be
rejected", so the guard-proof test (`test_contract_guard_proof.py`) can feed each
one to `assert_request_conforms` and prove the harness catches the regression it
exists to catch. If any of these ever stops failing, the guard has a hole.

`paths` are written as the CLI sent them (the `/v1` prefix is supplied by the
base_url at runtime; the validator tries both forms).

`reason` is the failure class we assert the validator detects:
  * "unregistered" — wrong method or path (405/404): no operation in the spec.
  * "extra"        — undeclared key under additionalProperties:false (422).
  * "missing"      — a required property absent from the body (422).
A single drift can trip more than one class; the test only requires that the
validator rejects it, and additionally asserts the message names the right class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.contract.conformance import MISSING, _Missing


@dataclass(frozen=True)
class Drift:
    label: str
    method: str
    path: str
    body: dict[str, Any] | None | _Missing
    reason: str  # primary failure class the message must mention
    note: str = ""
    # extra failure classes the same body also trips (for documentation only)
    also: tuple[str, ...] = field(default_factory=tuple)


# Order mirrors PR #12's "Files changed" walk-through.
HISTORICAL_DRIFTS: tuple[Drift, ...] = (
    Drift(
        label="assistants create without slug",
        method="POST",
        path="/assistants",
        body={"name": "Support", "instructions": "be helpful"},
        reason="missing",
        note="CreateAssistantRequest requires `slug` (+ `name`); slug was absent.",
    ),
    Drift(
        label="assistants update via PATCH (no PATCH handler)",
        method="PATCH",
        path="/assistants/asst_9",
        body={"name": "New", "instructions": "updated"},
        reason="unregistered",
        note="The gateway registers POST /assistants/{id}, never PATCH -> 405.",
    ),
    Drift(
        label="assistants run via /runs sub-resource with {input}",
        method="POST",
        path="/assistants/asst_42/runs",
        body={"input": "say hi"},
        reason="unregistered",
        note="No /assistants/{id}/runs route exists -> 404; run goes via "
        "/chat/completions with an asst_ model alias.",
    ),
    Drift(
        label="finetune create with flat forbidden keys",
        method="POST",
        path="/training/runs",
        body={"preset": "qwen32b-lora-r16", "dataset": "gpubox://ds",
              "name": "acme", "epochs": 3},
        reason="extra",
        note="GPUB-458: TrainingRunCreate is extra='forbid'; dataset/name/epochs "
        "are not declared.",
    ),
    Drift(
        label="training submit with flat forbidden keys",
        method="POST",
        path="/training/runs",
        body={"preset": "deberta-base", "dataset": "s3://bucket/x.jsonl",
              "name": "run-a", "epochs": 3},
        reason="extra",
        note="Same GPUB-458 forbidden body via the `training submit` surface.",
    ),
    Drift(
        label="hosting promote with legacy keys",
        method="POST",
        path="/hosting/models",
        body={"run_id": "run_abc", "tier": "warm", "name": "acme-v1"},
        reason="extra",
        note="HostedModelCreate is extra='forbid' and requires training_run_id/"
        "hosted_model_name/hosting_tier; run_id/tier/name are undeclared.",
        also=("missing",),
    ),
    Drift(
        label="hosting tier via PATCH with `tier` key",
        method="PATCH",
        path="/hosting/models/model_x",
        body={"tier": "always_hot"},
        reason="unregistered",
        note="Tier change is POST /hosting/models/{id}/transition with "
        "{hosting_tier}; there is no PATCH route and `tier` is a forbidden key.",
    ),
    Drift(
        label="training download via dead /artifact route",
        method="GET",
        path="/training/runs/run_d/artifact",
        body=MISSING,
        reason="unregistered",
        note="The /artifact route was never registered -> 404; downloads use the "
        "/download signed-URL envelope.",
    ),
)
