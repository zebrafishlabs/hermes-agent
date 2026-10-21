"""Empty model output still passes through the host's output transform."""
from types import SimpleNamespace
import logging
import pytest
from agent import turn_finalizer as F


@pytest.mark.parametrize("original", ["", None])
def test_empty_output_can_be_replaced_by_controlled_result(monkeypatch, original):
    calls = []
    def hook(name, logger, **kwargs):
        calls.append((name, kwargs))
        return ["[CRON_FAILURE]\nNo controlled receipt."]
    monkeypatch.setattr(F, "_invoke_hook_safely", hook)
    agent = SimpleNamespace(session_id="synthetic", model="synthetic", platform="cron")
    value, changed, prior = F.apply_llm_output_transform(agent, original, turn_id="turn", logger=logging.getLogger(__name__))
    assert value == "[CRON_FAILURE]\nNo controlled receipt."
    assert changed and prior == original
    assert len(calls) == 1
    assert calls[0][1]["response_text"] == ""
    assert F.apply_llm_output_transform(agent, value, turn_id="turn", logger=logging.getLogger(__name__))[0] == value
    assert len(calls) == 1


def test_empty_declined_transform_does_not_consume_later_text(monkeypatch):
    replies = iter([[], ["controlled later text"]])
    monkeypatch.setattr(F, "_invoke_hook_safely", lambda *a, **kw: next(replies))
    agent = SimpleNamespace(session_id="synthetic", model="synthetic", platform="slack")
    assert F.apply_llm_output_transform(agent, "", turn_id="turn", logger=logging.getLogger(__name__))[0] == ""
    assert F.apply_llm_output_transform(agent, "later", turn_id="turn", logger=logging.getLogger(__name__))[0] == "controlled later text"


@pytest.mark.parametrize("original", ["", None, "ordinary response"])
def test_no_transform_preserves_original(monkeypatch, original):
    monkeypatch.setattr(F, "_invoke_hook_safely", lambda *a, **kw: [])
    agent = SimpleNamespace(session_id="synthetic", model="synthetic", platform="slack")
    assert F.apply_llm_output_transform(agent, original, turn_id="turn", logger=logging.getLogger(__name__)) == (original, False, None)
