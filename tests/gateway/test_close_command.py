"""Legacy storage close APIs remain intact; /close itself is owned by the THR plugin."""
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_source(thread_id: str | None = "1234.5678") -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="u1",
        chat_id="C0123",
        user_name="tester",
        chat_type="group",
        thread_id=thread_id,
    )


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


def test_close_is_not_a_core_session_teardown_command():
    from gateway.run import GatewayRunner
    from hermes_cli.commands import resolve_command

    assert resolve_command("close") is None
    assert not hasattr(GatewayRunner, "_handle_close_command")
    assert "close" not in GatewayRunner._BUSY_SPECIAL_HANDLERS
