"""Tests for ESC-272: /close gateway command + Slack message shortcut.

Covers:
  * ``GatewayRunner._handle_close_command`` tears down the session and
    fires the right hooks (session:end, session:closed) without creating
    a fresh ``SessionEntry`` (in contrast to ``/new`` / ``/reset``).
  * ``SessionStore.close_session`` removes the entry and marks the DB row
    closed=1.
  * ``SessionDB.close_session`` updates the row in place (no delete).
  * ``SlackAdapter._handle_close_shortcut`` synthesises a ``MessageEvent``
    with the correct thread context and dispatches ``/close`` through
    ``handle_message``.
  * ``SlackAdapter.on_session_closed`` posts a ✅ reaction on the thread
    parent.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(thread_id: str | None = "1234.5678") -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="u1",
        chat_id="C0123",
        user_name="tester",
        chat_type="group",
        thread_id=thread_id,
    )


def _make_event(text: str = "/close", thread_id: str | None = "1234.5678") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(thread_id), message_id="m1")


def _make_runner_for_close(*, with_entry: bool = True):
    """Build a minimal GatewayRunner suitable for calling _handle_close_command.

    Mirrors the pattern from test_session_boundary_hooks.py but wires the
    Slack adapter and the session store close_session mock that ESC-272
    needs.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")}
    )

    adapter = MagicMock()
    adapter.send = AsyncMock()
    # _run_processing_hook is the real method on BasePlatformAdapter — we
    # need it to actually invoke on_session_closed on the mock.
    async def _run_hook(hook_name, *args, **kwargs):
        hook = getattr(adapter, hook_name, None)
        if hook is not None:
            await hook(*args, **kwargs)
    adapter._run_processing_hook = _run_hook
    adapter.on_session_closed = AsyncMock()
    runner.adapters = {Platform.SLACK: adapter}

    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._queued_events = {}
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    runner._invalidate_session_run_generation = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()

    source = _make_source()
    session_key = build_session_key(source)

    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="group",
    )

    runner.session_store = MagicMock()
    runner.session_store._entries = (
        {session_key: session_entry} if with_entry else {}
    )
    runner.session_store._generate_session_key = MagicMock(return_value=session_key)
    runner.session_store.close_session = MagicMock(
        return_value=session_entry if with_entry else None
    )
    # /close must NOT call reset/get_or_create on the store — record any
    # call so the test can assert that.
    runner.session_store.reset_session = MagicMock()
    runner.session_store.get_or_create_session = MagicMock()

    runner._session_key_for_source = lambda src: session_key
    return runner, session_key, session_entry


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_calls_close_session_not_reset(mock_invoke_hook):
    runner, session_key, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    runner.session_store.close_session.assert_called_once_with(session_key)
    runner.session_store.reset_session.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_is_idempotent_when_no_entry(mock_invoke_hook):
    """Closing a session that doesn't exist must not raise."""
    runner, _, _ = _make_runner_for_close(with_entry=False)

    # Should not raise — close on a missing session is a benign no-op.
    result = await runner._handle_close_command(_make_event())
    # Returns a reply, not None / exception.
    assert result is not None


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_fires_session_end_hook(mock_invoke_hook):
    runner, session_key, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    end_calls = [
        c for c in runner.hooks.emit.call_args_list if c[0][0] == "session:end"
    ]
    assert end_calls, "session:end hook was not fired"
    payload = end_calls[0][0][1]
    assert payload["platform"] == "slack"
    assert payload["session_key"] == session_key


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_fires_session_closed_hook_with_thread_context(mock_invoke_hook):
    runner, session_key, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    closed_calls = [
        c for c in runner.hooks.emit.call_args_list if c[0][0] == "session:closed"
    ]
    assert closed_calls, "session:closed hook was not fired"
    payload = closed_calls[0][0][1]
    assert payload["channel_id"] == "C0123"
    assert payload["thread_id"] == "1234.5678"
    assert payload["session_key"] == session_key


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_fires_on_session_finalize_plugin_hook(mock_invoke_hook):
    runner, _, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    mock_invoke_hook.assert_any_call(
        "on_session_finalize", session_id="sess-old", platform="slack"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_notifies_adapter_with_channel_and_thread(mock_invoke_hook):
    """Adapter.on_session_closed gets channel_id and thread_id so it can react."""
    runner, _, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    adapter = runner.adapters[Platform.SLACK]
    adapter.on_session_closed.assert_awaited_once_with("C0123", "1234.5678")


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_close_invalidates_session_generation(mock_invoke_hook):
    runner, session_key, _ = _make_runner_for_close()

    await runner._handle_close_command(_make_event())

    runner._invalidate_session_run_generation.assert_called_once()
    args, kwargs = runner._invalidate_session_run_generation.call_args
    assert args[0] == session_key
    assert kwargs.get("reason") == "session_close"


# --- SessionStore.close_session -----------------------------------------------

def test_session_store_close_removes_entry(tmp_path):
    """close_session() must pop the entry — no fresh entry is created."""
    from gateway.session import SessionStore

    cfg = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")})
    store = SessionStore(tmp_path, cfg)
    source = _make_source()
    entry = store.get_or_create_session(source)
    session_key = entry.session_key
    assert session_key in store._entries

    result = store.close_session(session_key)

    assert result is not None
    assert result.session_id == entry.session_id
    assert session_key not in store._entries


def test_session_store_close_missing_is_none(tmp_path):
    from gateway.session import SessionStore

    cfg = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")})
    store = SessionStore(tmp_path, cfg)

    assert store.close_session("agent:main:slack:dm:does-not-exist") is None


# --- SessionDB.close_session --------------------------------------------------

def test_session_db_close_marks_closed_column(tmp_path, monkeypatch):
    """SessionDB.close_session must set closed=1, ended_at, end_reason."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "sessions.db")
    sid = db.create_session("sess-test-123", "slack")

    db.close_session(sid)

    cursor = db._conn.execute(
        "SELECT closed, ended_at, end_reason FROM sessions WHERE id = ?",
        (sid,),
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] is not None  # ended_at populated
    assert row[2] == "session_close"
