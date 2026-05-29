"""Tests for the Linear platform-plugin adapter (ESC-286).

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_linear`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Coverage:
  - Linear-Signature HMAC-SHA256 validation (valid / bad / missing secret)
  - Inbound handler: signature gate, self-loop guard, mention filter
    (positive, negative, negative-lookahead @escher-hermes-fake), event-type
    filter, missing-issueId guard, happy-path 202 dispatch
  - Outbound: commentCreate GraphQL body/header shape, GraphQL-errors handling,
    success path
  - Plugin shape: register() wires the platform registry; the generic core
    hook (_plugin_route_registry) in gateway/platforms/webhook.py mounts the
    handler on the shared :8644 app.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_linear = load_plugin_adapter("linear")

LinearAdapter = _linear.LinearAdapter
check_requirements = _linear.check_requirements
register = _linear.register
_validate_linear_signature = _linear._validate_linear_signature
_post_linear_comment = _linear._post_linear_comment
_standalone_send = _linear._standalone_send
_env_enablement = _linear._env_enablement
_fetch_thread_context = _linear._fetch_thread_context
_build_thread_prompt = _linear._build_thread_prompt
_parse_chat_id = _linear._parse_chat_id
_MENTION_RE = _linear._MENTION_RE
_BOT_NAME = _linear._BOT_NAME
_BOT_EMAIL = _linear._BOT_EMAIL
LINEAR_SIG_HEADER = _linear.LINEAR_SIG_HEADER
LINEAR_GRAPHQL_URL = _linear.LINEAR_GRAPHQL_URL

_SECRET = "test-linear-signing-secret"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _linear_signature(body: bytes, secret: str) -> str:
    """Compute Linear-Signature: hex HMAC-SHA256 of the raw body."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _mock_request(headers=None, body=b""):
    """Lightweight mock aiohttp request (mirrors test_webhook_adapter)."""
    req = MagicMock()
    req.headers = headers or {}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _make_adapter(secret=_SECRET, api_key="lin_api_key"):
    cfg = PlatformConfig(
        enabled=True,
        extra={"api_key": api_key, "webhook_secret": secret},
    )
    return LinearAdapter(cfg)


def _comment_payload(body="@escher-hermes please help", actor_name="Chris",
                     actor_email="chris@imgix.com", issue_id="uuid-123",
                     identifier="ESC-99", title="Test issue", event_type="Comment",
                     parent_id=None, comment_id="comment-1"):
    data = {
        "id": comment_id,
        "body": body,
        "issueId": issue_id,
        "issue": {"identifier": identifier, "title": title},
    }
    if parent_id is not None:
        data["parentId"] = parent_id
    return {
        "type": event_type,
        "action": "create",
        "actor": {"id": "actor-1", "name": actor_name, "email": actor_email},
        "data": data,
        "url": f"https://linear.app/escher-graphics/issue/{identifier}",
    }


def _mock_graphql_session(*, status=200, json_body=None):
    """Module-level mock aiohttp.ClientSession context manager.

    Mirrors TestOutbound._mock_session for use by the read-back / thread tests.
    Returns (session, post_mock).
    """
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_body or {})
    resp.text = AsyncMock(return_value=json.dumps(json_body or {}))

    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(return_value=resp)
    post_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=post_ctx)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session, session.post


def _signed_request(payload, secret=_SECRET):
    raw = json.dumps(payload).encode()
    sig = _linear_signature(raw, secret)
    return _mock_request(headers={LINEAR_SIG_HEADER: sig}, body=raw)


async def _body(resp):
    """Extract the JSON dict from an aiohttp web.Response."""
    return json.loads(resp.body.decode())


# ---------------------------------------------------------------------------
# 1. Platform enum + requirements
# ---------------------------------------------------------------------------

def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform
    p = Platform("linear")
    assert p.value == "linear"
    assert Platform("linear") is p


def test_check_requirements_true_when_aiohttp_available():
    assert check_requirements() is True


def test_check_requirements_false_without_aiohttp(monkeypatch):
    monkeypatch.setattr(_linear, "AIOHTTP_AVAILABLE", False)
    assert check_requirements() is False
    monkeypatch.setattr(_linear, "AIOHTTP_AVAILABLE", True)


# ---------------------------------------------------------------------------
# 2. Signature validation
# ---------------------------------------------------------------------------

class TestSignatureValidation:

    def test_valid_signature_passes(self):
        body = b'{"hello":"world"}'
        sig = _linear_signature(body, _SECRET)
        assert _validate_linear_signature(body, _SECRET, sig) is True

    def test_bad_signature_fails(self):
        body = b'{"hello":"world"}'
        assert _validate_linear_signature(body, _SECRET, "deadbeef") is False

    def test_signature_over_different_body_fails(self):
        sig = _linear_signature(b'{"a":1}', _SECRET)
        assert _validate_linear_signature(b'{"a":2}', _SECRET, sig) is False

    def test_empty_secret_fails_closed(self):
        body = b'{"hello":"world"}'
        sig = _linear_signature(body, _SECRET)
        assert _validate_linear_signature(body, "", sig) is False

    def test_empty_sig_header_fails_closed(self):
        assert _validate_linear_signature(b"{}", _SECRET, "") is False

    def test_uppercase_hex_signature_still_matches_via_strip_only(self):
        # Linear sends lowercase hex; ensure we don't accidentally pass uppercase.
        body = b'{"hello":"world"}'
        sig = _linear_signature(body, _SECRET).upper()
        # compare_digest is case-sensitive — uppercase must NOT validate.
        assert _validate_linear_signature(body, _SECRET, sig) is False


# ---------------------------------------------------------------------------
# 3. Mention regex
# ---------------------------------------------------------------------------

class TestMentionRegex:

    @pytest.mark.parametrize("text", [
        "@escher-hermes help",
        "hey @Escher-Hermes can you",
        "ping @ESCHER-HERMES",
        "trailing @escher-hermes.",
        "@escher-hermes, please",
    ])
    def test_matches(self, text):
        assert _MENTION_RE.search(text) is not None

    @pytest.mark.parametrize("text", [
        "plain comment no mention",
        "talking about escher-hermes without at-sign",
        "@escher-hermes-fake should not match",
        "@escher-hermesbot extra chars",
        "email escher-hermes+linear@imgix.com in prose",
    ])
    def test_no_match(self, text):
        assert _MENTION_RE.search(text) is None


# ---------------------------------------------------------------------------
# 4. Inbound handler
# ---------------------------------------------------------------------------

class TestInboundHandler:

    def test_bad_signature_returns_401(self):
        adapter = _make_adapter()
        payload = _comment_payload()
        raw = json.dumps(payload).encode()
        req = _mock_request(headers={LINEAR_SIG_HEADER: "wrong"}, body=raw)
        resp = _run(adapter._handle_linear_webhook(req))
        assert resp.status == 401

    def test_missing_secret_returns_403(self):
        adapter = _make_adapter(secret="")
        payload = _comment_payload()
        raw = json.dumps(payload).encode()
        # even a "valid" sig can't help — secret unset means fail-closed
        req = _mock_request(headers={LINEAR_SIG_HEADER: "anything"}, body=raw)
        resp = _run(adapter._handle_linear_webhook(req))
        assert resp.status == 403

    def test_non_comment_event_ignored(self):
        adapter = _make_adapter()
        payload = _comment_payload(event_type="Issue")
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "event_type"

    def test_self_loop_guard_by_name(self):
        adapter = _make_adapter()
        payload = _comment_payload(actor_name=_BOT_NAME, actor_email="x@y.z")
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "self_loop"

    def test_self_loop_guard_by_email(self):
        adapter = _make_adapter()
        payload = _comment_payload(actor_name="Someone", actor_email=_BOT_EMAIL)
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "self_loop"

    def test_no_mention_ignored(self):
        adapter = _make_adapter()
        payload = _comment_payload(body="just a regular comment")
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "no_mention"

    def test_fake_mention_ignored(self):
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes-fake hi")
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "no_mention"

    def test_missing_issue_id_returns_400(self):
        adapter = _make_adapter()
        payload = _comment_payload()
        payload["data"]["issueId"] = ""
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 400

    def test_happy_path_dispatches_and_returns_202(self):
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes status?")
        with patch.object(adapter, "handle_message", new=AsyncMock()) as hm, \
             patch.object(_linear, "_fetch_thread_context",
                          new=AsyncMock(return_value=None)):
            # let the create_task'd coroutine run
            resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        body = _run(_body(resp))
        assert body["status"] == "accepted"
        assert body["issue"] == "ESC-99"
        hm.assert_called_once()
        # the issue metadata should be cached for the outbound bulletin
        assert adapter._issue_cache["uuid-123"]["identifier"] == "ESC-99"

    def test_bad_json_returns_400(self):
        adapter = _make_adapter()
        raw = b"not json{"
        sig = _linear_signature(raw, _SECRET)
        req = _mock_request(headers={LINEAR_SIG_HEADER: sig}, body=raw)
        resp = _run(adapter._handle_linear_webhook(req))
        assert resp.status == 400


# ---------------------------------------------------------------------------
# 5. Outbound: commentCreate
# ---------------------------------------------------------------------------

class TestOutbound:

    def _mock_session(self, *, status=200, json_body=None):
        """Build a mock aiohttp.ClientSession context manager."""
        resp = MagicMock()
        resp.status = status
        resp.json = AsyncMock(return_value=json_body or {})
        resp.text = AsyncMock(return_value=json.dumps(json_body or {}))

        post_ctx = MagicMock()
        post_ctx.__aenter__ = AsyncMock(return_value=resp)
        post_ctx.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=post_ctx)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session, session.post

    def test_comment_create_success_shape(self):
        ok = {"data": {"commentCreate": {"success": True,
              "comment": {"id": "c1", "url": "https://linear.app/c/c1"}}}}
        session, post = self._mock_session(json_body=ok)
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hello", "api_key"))
        assert result["success"] is True
        assert result["comment_id"] == "c1"
        assert result["comment_url"] == "https://linear.app/c/c1"
        # Verify the GraphQL request shape
        _, kwargs = post.call_args
        assert kwargs["headers"]["Authorization"] == "api_key"  # no Bearer prefix
        assert "Bearer" not in kwargs["headers"]["Authorization"]
        assert kwargs["json"]["variables"] == {"issueId": "uuid-1", "body": "hello"}
        assert "commentCreate" in kwargs["json"]["query"]

    def test_comment_create_graphql_errors(self):
        err = {"errors": [{"message": "Issue not found"}]}
        session, _ = self._mock_session(json_body=err)
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hi", "api_key"))
        assert result["success"] is False
        assert "Issue not found" in result["error"]

    def test_comment_create_http_error(self):
        session, _ = self._mock_session(status=500, json_body={})
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hi", "api_key"))
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    def test_comment_create_success_false(self):
        body = {"data": {"commentCreate": {"success": False, "comment": None}}}
        session, _ = self._mock_session(json_body=body)
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hi", "api_key"))
        assert result["success"] is False

    def test_send_posts_comment_and_bulletin(self):
        adapter = _make_adapter()
        adapter._issue_cache["uuid-9"] = {"identifier": "ESC-9", "title": "Foo"}
        with patch.object(_linear, "_post_linear_comment",
                          new=AsyncMock(return_value={"success": True,
                                                      "comment_id": "c2",
                                                      "comment_url": "u"})) as pc, \
             patch.object(_linear, "_post_slack_summary", new=AsyncMock()) as sb:
            result = _run(adapter.send("linear:uuid-9", "Done.\nSecond line"))
        assert result.success is True
        assert result.message_id == "c2"
        pc.assert_called_once_with("uuid-9", "Done.\nSecond line", "lin_api_key",
                                   parent_id=None)
        # bulletin is a one-liner: "ESC-9 Foo: Done."
        sb.assert_called_once()
        bulletin = sb.call_args[0][0]
        assert bulletin.startswith("ESC-9 Foo:")
        assert "Second line" not in bulletin

    def test_summary_channel_defaults_to_escher_linear(self, monkeypatch):
        monkeypatch.delenv("LINEAR_SLACK_CHANNEL", raising=False)
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        sent = AsyncMock(return_value={"success": True})
        with patch("tools.send_message_tool._send_slack", new=sent):
            _run(_linear._post_slack_summary("ESC-1: hi"))
        # default channel is #escher-linear
        assert sent.call_args[0][1] == _linear._DEFAULT_SLACK_CHANNEL == "C0B6KMBPAGZ"

    def test_summary_channel_env_override(self, monkeypatch):
        monkeypatch.setenv("LINEAR_SLACK_CHANNEL", "C0OVERRIDE")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        sent = AsyncMock(return_value={"success": True})
        with patch("tools.send_message_tool._send_slack", new=sent):
            _run(_linear._post_slack_summary("ESC-1: hi"))
        assert sent.call_args[0][1] == "C0OVERRIDE"

    def test_summary_skipped_without_token(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        sent = AsyncMock()
        with patch("tools.send_message_tool._send_slack", new=sent):
            _run(_linear._post_slack_summary("ESC-1: hi"))
        sent.assert_not_called()

    def test_send_empty_issue_id_fails(self):
        adapter = _make_adapter()
        result = _run(adapter.send("linear:", "hi"))
        assert result.success is False

    def test_standalone_send_uses_api_key(self, monkeypatch):
        monkeypatch.setenv("LINEAR_API_KEY", "env_key")
        with patch.object(_linear, "_post_linear_comment",
                          new=AsyncMock(return_value={"success": True})) as pc:
            _run(_standalone_send(MagicMock(extra={}), "linear:uuid-x", "msg"))
        pc.assert_called_once_with("uuid-x", "msg", "env_key", parent_id=None)


# ---------------------------------------------------------------------------
# 6. Plugin shape + generic core hook
# ---------------------------------------------------------------------------

class TestPluginShape:

    def test_register_calls_register_platform(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "linear"
        assert kwargs["label"] == "Linear"
        assert "LINEAR_API_KEY" in kwargs["required_env"]
        assert kwargs["standalone_sender_fn"] is _standalone_send

    def test_register_is_exported_from_package_init(self):
        """The plugin loader imports register() from the package __init__, not
        adapter.py. A bare __init__ (no re-export) makes the loader log
        'has no register() function' and the platform never registers — the
        exact bug that blocked the first deploy. Guard the re-export by reading
        the source (avoids relative-import context issues in the test harness)."""
        from pathlib import Path
        init_path = Path(_linear.__file__).resolve().parent / "__init__.py"
        src = init_path.read_text()
        assert "register" in src and "import register" in src, (
            "plugins/platforms/linear/__init__.py must re-export register() "
            "(e.g. 'from .adapter import register')"
        )

    def test_connect_registers_route_in_core_hook(self):
        from gateway.platforms.webhook import _plugin_route_registry
        _plugin_route_registry.pop("/webhooks/linear-comments", None)
        adapter = _make_adapter()
        _run(adapter.connect())
        assert "/webhooks/linear-comments" in _plugin_route_registry
        # disconnect removes it (no stale-handler accumulation)
        _run(adapter.disconnect())
        assert "/webhooks/linear-comments" not in _plugin_route_registry

    def test_wildcard_delegates_to_plugin_route_regardless_of_connect_order(self):
        """Order-independence (ESC-286): even if the webhook adapter built its
        app BEFORE the linear plugin connected (so the mount-time injection
        loop never saw the route), the generic /webhooks/{route_name} handler
        must still delegate to the plugin handler at request time. This is the
        correctness guarantee that mount-time injection alone does NOT provide.
        """
        from gateway.platforms.webhook import WebhookAdapter, _plugin_route_registry
        from gateway.config import PlatformConfig

        called = {}

        async def fake_handler(request):
            from aiohttp import web
            called["hit"] = True
            return web.json_response({"status": "linear-handled"}, status=202)

        # Webhook adapter connected FIRST: registry is empty at mount time.
        wh = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "x"}))
        # Linear plugin connects AFTER — registers its route now.
        _plugin_route_registry["/webhooks/linear-comments"] = fake_handler
        try:
            req = _mock_request(body=b"{}")
            req.match_info = {"route_name": "linear-comments"}
            resp = _run(wh._handle_webhook(req))
            assert called.get("hit") is True
            assert resp.status == 202
        finally:
            _plugin_route_registry.pop("/webhooks/linear-comments", None)

    def test_env_enablement_seeds_extra(self, monkeypatch):
        monkeypatch.setenv("LINEAR_API_KEY", "k")
        monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "s")
        out = _env_enablement()
        assert out["api_key"] == "k"
        assert out["webhook_secret"] == "s"

    def test_env_enablement_none_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LINEAR_API_KEY", raising=False)
        assert _env_enablement() is None


# ---------------------------------------------------------------------------
# 7. Thread-per-comment sessions + parentId threading (ESC-290)
# ---------------------------------------------------------------------------

class TestThreadSessions:
    """Root-comment resolution and 3-part chat_id derivation."""

    def test_top_level_comment_root_is_own_id(self):
        # A top-level comment has no parentId -> root is the comment's own id.
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes hi", comment_id="cmt-top")
        captured = {}

        async def _capture(event):
            captured["event"] = event

        with patch.object(adapter, "handle_message", new=_capture), \
             patch.object(_linear, "_fetch_thread_context",
                          new=AsyncMock(return_value=None)):
            resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        chat_id = captured["event"].source.chat_id
        # root == own id since no parentId
        assert chat_id == "linear:uuid-123:cmt-top"

    def test_reply_comment_root_is_parent_id(self):
        # A reply carries parentId -> root is the parent (the thread root).
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes hi",
                                   comment_id="cmt-reply", parent_id="cmt-root")
        captured = {}

        async def _capture(event):
            captured["event"] = event

        with patch.object(adapter, "handle_message", new=_capture), \
             patch.object(_linear, "_fetch_thread_context",
                          new=AsyncMock(return_value=None)):
            resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        chat_id = captured["event"].source.chat_id
        assert chat_id == "linear:uuid-123:cmt-root"

    def test_session_chat_id_contains_issue_and_root(self):
        issue_id, root = _parse_chat_id("linear:uuid-123:cmt-root")
        assert issue_id == "uuid-123"
        assert root == "cmt-root"

    def test_parse_legacy_two_part_chat_id(self):
        issue_id, root = _parse_chat_id("linear:uuid-123")
        assert issue_id == "uuid-123"
        assert root is None

    def test_parse_bare_issue_id(self):
        issue_id, root = _parse_chat_id("uuid-123")
        assert issue_id == "uuid-123"
        assert root is None


class TestParentIdThreading:
    """commentCreate parentId variable handling + send() pass-through."""

    def _mock_session(self, *, status=200, json_body=None):
        return _mock_graphql_session(status=status, json_body=json_body)

    def test_post_comment_includes_parent_id_when_given(self):
        ok = {"data": {"commentCreate": {"success": True,
              "comment": {"id": "c1", "url": "u"}}}}
        session, post = self._mock_session(json_body=ok)
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hi", "api_key",
                                               parent_id="root-c"))
        assert result["success"] is True
        _, kwargs = post.call_args
        assert kwargs["json"]["variables"]["parentId"] == "root-c"
        assert "$parentId" in kwargs["json"]["query"]

    def test_post_comment_omits_parent_id_when_none(self):
        ok = {"data": {"commentCreate": {"success": True,
              "comment": {"id": "c1", "url": "u"}}}}
        session, post = self._mock_session(json_body=ok)
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", "hi", "api_key"))
        assert result["success"] is True
        _, kwargs = post.call_args
        # parentId omitted from variables -> GraphQL treats it as null (top-level)
        assert "parentId" not in kwargs["json"]["variables"]

    def test_send_three_part_chat_id_threads_under_root(self):
        adapter = _make_adapter()
        with patch.object(_linear, "_post_linear_comment",
                          new=AsyncMock(return_value={"success": True,
                                                      "comment_id": "c2"})) as pc, \
             patch.object(_linear, "_post_slack_summary", new=AsyncMock()):
            result = _run(adapter.send("linear:uuid-9:root-7", "Done."))
        assert result.success is True
        pc.assert_called_once_with("uuid-9", "Done.", "lin_api_key",
                                   parent_id="root-7")

    def test_send_legacy_two_part_chat_id_top_level(self):
        adapter = _make_adapter()
        with patch.object(_linear, "_post_linear_comment",
                          new=AsyncMock(return_value={"success": True,
                                                      "comment_id": "c2"})) as pc, \
             patch.object(_linear, "_post_slack_summary", new=AsyncMock()):
            result = _run(adapter.send("linear:uuid-9", "Done."))
        assert result.success is True
        pc.assert_called_once_with("uuid-9", "Done.", "lin_api_key",
                                   parent_id=None)

    def test_standalone_send_three_part_threads(self, monkeypatch):
        monkeypatch.setenv("LINEAR_API_KEY", "env_key")
        with patch.object(_linear, "_post_linear_comment",
                          new=AsyncMock(return_value={"success": True})) as pc:
            _run(_standalone_send(MagicMock(extra={}), "linear:uuid-x:root-y", "msg"))
        pc.assert_called_once_with("uuid-x", "msg", "env_key", parent_id="root-y")


class TestThreadReadback:
    """GraphQL read-back of issue + full thread, and prompt assembly."""

    def _readback_body(self):
        return {"data": {
            "issue": {
                "identifier": "ESC-99",
                "title": "Test issue",
                "description": "The issue body explains the goal.",
                "state": {"name": "In Progress"},
            },
            "comment": {
                "id": "cmt-root",
                "body": "Earlier pre-mention comment from a teammate.",
                "createdAt": "2026-05-01T00:00:00Z",
                "user": {"name": "Dana", "displayName": "Dana D"},
                "children": {"nodes": [
                    {"id": "cmt-reply",
                     "body": "@escher-hermes please summarize",
                     "createdAt": "2026-05-02T00:00:00Z",
                     "user": {"name": "Chris", "displayName": "Chris C"}},
                ]},
            },
        }}

    def test_fetch_thread_context_parses_issue_and_thread(self):
        session, _ = _mock_graphql_session(json_body=self._readback_body())
        with patch("aiohttp.ClientSession", return_value=session):
            ctx = _run(_fetch_thread_context("uuid-123", "cmt-root", "api_key"))
        assert ctx["issue"]["status"] == "In Progress"
        assert ctx["issue"]["identifier"] == "ESC-99"
        bodies = [c["body"] for c in ctx["thread"]]
        assert "Earlier pre-mention comment from a teammate." in bodies
        assert "@escher-hermes please summarize" in bodies
        # chronological: root first, then reply
        assert ctx["thread"][0]["author"] == "Dana D"

    def test_fetch_thread_context_trims_description(self):
        body = self._readback_body()
        body["data"]["issue"]["description"] = "x" * 5000
        session, _ = _mock_graphql_session(json_body=body)
        with patch("aiohttp.ClientSession", return_value=session):
            ctx = _run(_fetch_thread_context("uuid-123", "cmt-root", "api_key"))
        assert len(ctx["issue"]["description"]) < 5000
        assert ctx["issue"]["description"].endswith("…[truncated]")

    def test_inbound_prompt_includes_thread_and_status(self):
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes please summarize",
                                   comment_id="cmt-reply", parent_id="cmt-root",
                                   actor_name="Chris")
        captured = {}

        async def _capture(event):
            captured["event"] = event

        session, _ = _mock_graphql_session(json_body=self._readback_body())
        with patch.object(adapter, "handle_message", new=_capture), \
             patch("aiohttp.ClientSession", return_value=session):
            resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        text = captured["event"].text
        # issue header + status present
        assert "ESC-99" in text
        assert "In Progress" in text
        # the pre-mention comment is included (not truncated at the mention)
        assert "Earlier pre-mention comment from a teammate." in text
        # the triggering comment is highlighted
        assert "please summarize" in text
        # commenter surfaced for @-mention guidance
        assert "Chris" in text

    def test_inbound_readback_failure_falls_back(self):
        adapter = _make_adapter()
        payload = _comment_payload(body="@escher-hermes help",
                                   comment_id="cmt-x")
        captured = {}

        async def _capture(event):
            captured["event"] = event

        # _fetch_thread_context returns None (read-back failed) -> fallback.
        with patch.object(adapter, "handle_message", new=_capture), \
             patch.object(_linear, "_fetch_thread_context",
                          new=AsyncMock(return_value=None)):
            resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
            _run(asyncio.sleep(0))
        # Still dispatched an agent run despite read-back failure.
        assert resp.status == 202
        assert "event" in captured
        text = captured["event"].text
        assert "ESC-99" in text
        assert "@escher-hermes help" in text

    def test_fetch_thread_context_http_error_returns_none(self):
        session, _ = _mock_graphql_session(status=500, json_body={})
        with patch("aiohttp.ClientSession", return_value=session):
            ctx = _run(_fetch_thread_context("uuid-123", "cmt-root", "api_key"))
        assert ctx is None


class TestMentionOwner:
    """@human mention preserved outbound; self-loop guard still drops bot events."""

    def test_at_human_mention_preserved_in_outbound_body(self):
        ok = {"data": {"commentCreate": {"success": True,
              "comment": {"id": "c1", "url": "u"}}}}
        session, post = _mock_graphql_session(json_body=ok)
        body = "@Chris C can you confirm the rollout window?"
        with patch("aiohttp.ClientSession", return_value=session):
            result = _run(_post_linear_comment("uuid-1", body, "api_key",
                                               parent_id="root-c"))
        assert result["success"] is True
        _, kwargs = post.call_args
        # verbatim — the @human mention is not stripped or altered
        assert kwargs["json"]["variables"]["body"] == body

    def test_self_loop_guard_still_drops_bot_events(self):
        # An @human mention in the bot's reply does not change the inbound
        # actor; bot-actored events are still dropped (no re-wake loop).
        adapter = _make_adapter()
        payload = _comment_payload(actor_name=_BOT_NAME,
                                   body="@Chris C done — over to you")
        resp = _run(adapter._handle_linear_webhook(_signed_request(payload)))
        assert resp.status == 200
        assert _run(_body(resp))["reason"] == "self_loop"


class TestThreadResolution:
    """ESC-291: stay quiet in resolved threads; respond on open / re-open.

    The thread's root comment carries ``resolvedAt`` — non-null ⇒ resolved
    (suppress), null/absent ⇒ open or re-opened (respond).  Read-back failure
    is fail-OPEN (respond), since resolution status is then indeterminate.
    """

    def _readback_body(self, *, resolved_at=None):
        """Read-back payload with the root comment's resolvedAt set or cleared."""
        return {"data": {
            "issue": {
                "identifier": "ESC-99",
                "title": "Test issue",
                "description": "The issue body explains the goal.",
                "state": {"name": "In Progress"},
            },
            "comment": {
                "id": "cmt-root",
                "body": "Root comment of the thread.",
                "createdAt": "2026-05-01T00:00:00Z",
                "resolvedAt": resolved_at,
                "user": {"name": "Dana", "displayName": "Dana D"},
                "children": {"nodes": [
                    {"id": "cmt-reply",
                     "body": "@escher-hermes please help",
                     "createdAt": "2026-05-02T00:00:00Z",
                     "resolvedAt": None,
                     "user": {"name": "Chris", "displayName": "Chris C"}},
                ]},
            },
        }}

    def _mention_payload(self):
        return _comment_payload(body="@escher-hermes please help",
                                comment_id="cmt-reply", parent_id="cmt-root",
                                actor_name="Chris")

    # --- _fetch_thread_context parses resolvedAt into the resolved bool ------

    def test_fetch_context_resolved_true_when_resolvedAt_set(self):
        body = self._readback_body(resolved_at="2026-05-03T00:00:00Z")
        session, _ = _mock_graphql_session(json_body=body)
        with patch("aiohttp.ClientSession", return_value=session):
            ctx = _run(_fetch_thread_context("uuid-123", "cmt-root", "api_key"))
        assert ctx["resolved"] is True

    def test_fetch_context_resolved_false_when_resolvedAt_null(self):
        body = self._readback_body(resolved_at=None)
        session, _ = _mock_graphql_session(json_body=body)
        with patch("aiohttp.ClientSession", return_value=session):
            ctx = _run(_fetch_thread_context("uuid-123", "cmt-root", "api_key"))
        assert ctx["resolved"] is False

    # --- inbound gate: resolved suppresses, open/re-open dispatches ----------

    def test_resolved_thread_suppresses_dispatch(self):
        adapter = _make_adapter()
        captured = {}

        async def _capture(event):
            captured["event"] = event

        body = self._readback_body(resolved_at="2026-05-03T00:00:00Z")
        session, _ = _mock_graphql_session(json_body=body)
        with patch.object(adapter, "handle_message", new=_capture), \
             patch("aiohttp.ClientSession", return_value=session):
            resp = _run(adapter._handle_linear_webhook(
                _signed_request(self._mention_payload())))
            _run(asyncio.sleep(0))
        # Agent NOT invoked; ignored with the resolved reason.
        assert "event" not in captured
        assert _run(_body(resp))["reason"] == "thread_resolved"

    def test_open_thread_dispatches(self):
        adapter = _make_adapter()
        captured = {}

        async def _capture(event):
            captured["event"] = event

        body = self._readback_body(resolved_at=None)
        session, _ = _mock_graphql_session(json_body=body)
        with patch.object(adapter, "handle_message", new=_capture), \
             patch("aiohttp.ClientSession", return_value=session):
            resp = _run(adapter._handle_linear_webhook(
                _signed_request(self._mention_payload())))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        assert "event" in captured

    def test_reopened_thread_dispatches(self):
        # A re-opened thread is structurally identical to a never-resolved one
        # (resolvedAt cleared back to null) — assert explicitly to document the
        # re-open path uses the same gate with no special handling.
        adapter = _make_adapter()
        captured = {}

        async def _capture(event):
            captured["event"] = event

        body = self._readback_body(resolved_at=None)  # was resolved, now re-opened
        session, _ = _mock_graphql_session(json_body=body)
        with patch.object(adapter, "handle_message", new=_capture), \
             patch("aiohttp.ClientSession", return_value=session):
            resp = _run(adapter._handle_linear_webhook(
                _signed_request(self._mention_payload())))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        assert "event" in captured

    def test_readback_failure_fails_open_and_dispatches(self):
        # context is None (read-back failed) -> resolution indeterminate ->
        # respond rather than silently swallow the mention.
        adapter = _make_adapter()
        captured = {}

        async def _capture(event):
            captured["event"] = event

        with patch.object(adapter, "handle_message", new=_capture), \
             patch.object(_linear, "_fetch_thread_context",
                          new=AsyncMock(return_value=None)):
            resp = _run(adapter._handle_linear_webhook(
                _signed_request(self._mention_payload())))
            _run(asyncio.sleep(0))
        assert resp.status == 202
        assert "event" in captured
