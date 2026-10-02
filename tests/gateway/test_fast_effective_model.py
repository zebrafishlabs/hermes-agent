"""/fast must validate the model that this conversation will actually use."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.gateway.test_fast_command import _make_runner


@pytest.fixture
def fast_context(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"default": "claude-fable-5-1"},
                                    "agent": {"service_tier": "normal"}}))
    runner = _make_runner()
    source = SessionSource(platform=Platform.SLACK, chat_id="channel", chat_type="group",
                           user_id="user", thread_id="thread")
    event = MessageEvent(text="/fast ultrafast", source=source, message_id="m1")
    return runner, event, path


@pytest.mark.asyncio
@pytest.mark.parametrize("global_model,channel_model,session_model,accepted", [
    ("claude-fable-5-1", None, "gpt-6-astra", True),
    ("gpt-6-astra", None, "claude-fable-5-1", False),
    ("claude-fable-5-1", "gpt-6-astra", None, True),
    ("gpt-6-astra", "claude-fable-5-1", None, False),
    ("claude-fable-5-1", "gpt-6-astra", "gpt-6-sol", False),
    ("gpt-6-sol", "claude-fable-5-1", "gpt-6-astra", True),
    ("gpt-6-astra", None, None, True),
])
async def test_ultrafast_uses_session_then_channel_then_global(
    fast_context, global_model, channel_model, session_model, accepted,
):
    runner, event, path = fast_context
    config = yaml.safe_load(path.read_text())
    config["model"]["default"] = global_model
    path.write_text(yaml.safe_dump(config))
    original = path.read_bytes()
    key = runner._session_key_for_source(event.source)
    if channel_model:
        runner.config.platforms = {
            event.source.platform: SimpleNamespace(channel_overrides={
                event.source.chat_id: SimpleNamespace(model=channel_model),
            }),
        }
    if session_model:
        runner._session_model_overrides[key] = {"model": session_model}

    response = await runner._handle_fast_command(event)

    if accepted:
        assert "ULTRAFAST" in response
        assert runner._resolve_session_service_tier(session_key=key) == "ultrafast"
        route = runner._resolve_turn_agent_config("hello", "gpt-6-astra", {
            "provider": "openai-api", "base_url": "https://api.openai.com/v1",
            "api_mode": "codex_responses", "command": None, "args": [],
            "credential_pool": None,
        })
        assert route["request_overrides"] == {"service_tier": "ultrafast"}
    else:
        assert "only available" in response
        assert runner._resolve_session_service_tier(session_key=key) is None
        assert key not in runner._session_service_tier_overrides
    assert runner._resolve_session_service_tier(session_key="another-thread") is None
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_fast_status_and_picker_use_the_session_model(fast_context):
    runner, event, path = fast_context
    key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[key] = {"model": "gpt-6-astra"}
    runner._try_send_choice_picker = AsyncMock(return_value=False)

    selected = await runner._handle_fast_command(replace(event, text="/fast fast"))
    assert "FAST" in selected
    status = await runner._handle_fast_command(replace(event, text="/fast status"))
    assert "Current mode: `fast`" in status
    picker = runner._try_send_choice_picker.call_args.kwargs
    assert "ultrafast" in {item["value"] for item in picker["choices"]}
    await picker["on_choice_selected"](event.source.chat_id, "ultrafast")
    assert runner._resolve_session_service_tier(session_key=key) == "ultrafast"
    assert yaml.safe_load(path.read_text())["agent"]["service_tier"] == "normal"


@pytest.mark.asyncio
async def test_fast_rehydrates_persisted_session_model(fast_context):
    runner, event, _ = fast_context
    key = runner._session_key_for_source(event.source)
    # Exercise real rehydration without credentials or provider network calls.
    runner.session_store.get_model_override = lambda candidate: (
        {"model": "gpt-6-astra"} if candidate == key else None
    )

    response = await runner._handle_fast_command(event)

    assert "ULTRAFAST" in response
    assert runner._session_model_override(key)["model"] == "gpt-6-astra"
    assert runner._resolve_session_service_tier(session_key=key) == "ultrafast"


@pytest.mark.asyncio
async def test_fast_stores_tier_under_normalized_topic_key(fast_context, monkeypatch):
    runner, event, _ = fast_context
    source = replace(event.source, platform=Platform.TELEGRAM, chat_type="dm", thread_id=None)
    event = replace(event, source=source)
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda source: "topic")
    normalized = replace(source, thread_id="topic")
    key = runner._session_key_for_source(normalized)
    raw_key = runner._session_key_for_source(source)
    assert key != raw_key
    runner._session_model_overrides[key] = {"model": "gpt-6-astra"}

    response = await runner._handle_fast_command(event)

    assert "ULTRAFAST" in response
    assert runner._resolve_session_service_tier(session_key=key) == "ultrafast"
    assert raw_key not in runner._session_service_tier_overrides
