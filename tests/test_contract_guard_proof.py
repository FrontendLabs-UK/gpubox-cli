"""Guard-proof: the harness actually rejects the eight historical drifts.

A conformance harness that passes everything is worthless. This file feeds each
of the eight PR-#12-era bad request shapes to `assert_request_conforms` and
asserts each one FAILS — proving the validator catches the regressions it exists
to catch, and that its error message names the right failure class.

It also asserts the validator ACCEPTS the corrected shape for each drift, so the
guard is not just rejecting everything (a stuck-closed validator would be equally
useless). Together these two directions pin the validator's decision boundary.
"""

from __future__ import annotations

import pytest

from tests.contract.conformance import (
    MISSING,
    ConformanceError,
    assert_request_conforms,
    load_spec,
)
from tests.contract.historical_drifts import HISTORICAL_DRIFTS, Drift

SPEC = load_spec()

# The class -> phrase the validator emits, so we can assert the RIGHT reason.
_REASON_PHRASE = {
    "unregistered": "unregistered method/path",
    "extra": "undeclared key",
    "missing": "missing required",
}


def test_there_are_exactly_eight_historical_drifts() -> None:
    """Locks the count at 8 so the fixture set cannot silently shrink."""
    assert len(HISTORICAL_DRIFTS) == 8


@pytest.mark.parametrize("drift", HISTORICAL_DRIFTS, ids=lambda d: d.label)
def test_validator_rejects_historical_drift(drift: Drift) -> None:
    """Every historical bad body must raise ConformanceError."""
    with pytest.raises(ConformanceError) as exc_info:
        assert_request_conforms(SPEC, drift.method, drift.path, drift.body)
    message = str(exc_info.value)
    expected_phrase = _REASON_PHRASE[drift.reason]
    assert expected_phrase in message, (
        f"{drift.label}: expected the {drift.reason!r} failure class "
        f"({expected_phrase!r}) in the message, got: {message}"
    )


# Additional edge-case rejections the validator must enforce (not in the original
# eight, but the same top-level-shape failure class Codex flagged in review).
_EDGE_REJECTIONS = [
    # No body at all to an op whose JSON requestBody is required.
    ("missing required body", "POST", "/assistants", MISSING, "requires a JSON"),
    # A present-but-empty object to a forbid+required schema (missing `slug`/`name`).
    ("empty object body", "POST", "/assistants", {}, "missing required"),
    # A present non-object body (bare list/string/null) to an object schema.
    ("non-object body: list", "POST", "/assistants", [], "expected a JSON object"),
    ("non-object body: string", "POST", "/assistants", "oops", "expected a JSON object"),
    ("non-object body: null", "POST", "/assistants", None, "expected a JSON object"),
    # Any body (even {}) to a body-less operation.
    ("body to body-less op", "POST", "/argus/agents/a1/run", {}, "declares no JSON"),
]


@pytest.mark.parametrize(
    "label,method,path,body,phrase", _EDGE_REJECTIONS, ids=lambda v: v if isinstance(v, str) else ""
)
def test_validator_rejects_edge_shape(
    label: str, method: str, path: str, body: object, phrase: str
) -> None:
    """Top-level shape edge cases that must NOT false-pass."""
    with pytest.raises(ConformanceError) as exc_info:
        assert_request_conforms(SPEC, method, path, body)
    assert phrase in str(exc_info.value), (
        f"{label}: expected {phrase!r} in message, got: {exc_info.value}"
    )


# The corrected shape for each drift — the validator must ACCEPT these, proving
# the harness is not stuck-closed (rejecting everything).
_CORRECTED = [
    ("assistants create (fixed)", "POST", "/assistants",
     {"slug": "support", "name": "Support", "instructions": "be helpful"}),
    ("assistants update (fixed: POST)", "POST", "/assistants/asst_9",
     {"name": "New", "instructions": "updated"}),
    ("assistants run (fixed: /chat/completions)", "POST", "/chat/completions",
     {"model": "asst_42", "messages": [{"role": "user", "content": "say hi"}]}),
    ("finetune create (fixed: preset + hyperparams)", "POST", "/training/runs",
     {"preset": "qwen32b-lora-r16",
      "hyperparams": {"epochs": 3, "batch_size": 4, "learning_rate": 0.0001}}),
    ("training submit (fixed: preset only)", "POST", "/training/runs",
     {"preset": "deberta-base"}),
    ("hosting promote (fixed keys)", "POST", "/hosting/models",
     {"training_run_id": "run_abc", "hosted_model_name": "acme-v1",
      "hosting_tier": "warm"}),
    ("hosting tier (fixed: /transition POST)", "POST",
     "/hosting/models/model_x/transition", {"hosting_tier": "always_hot"}),
    ("training download (fixed: /download envelope)", "GET",
     "/training/runs/run_d/download", MISSING),
]


@pytest.mark.parametrize(
    "label,method,path,body", _CORRECTED, ids=lambda v: v if isinstance(v, str) else ""
)
def test_validator_accepts_corrected_shape(
    label: str, method: str, path: str, body: object
) -> None:
    """The post-PR-#12 corrected shape must pass — guard is not stuck-closed."""
    # Raises if it (wrongly) rejects a valid shape.
    assert_request_conforms(SPEC, method, path, body)
