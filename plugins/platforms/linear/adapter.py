"""
Linear platform adapter for Hermes Agent.

Receives webhook events from Linear's /webhooks/linear-comments path on the
shared :8644 ingress (gateway/platforms/webhook.py extra_route_handlers /
_plugin_route_registry).  Does NOT run a second HTTP server.

Security:
  - Validates ``Linear-Signature: <hex>`` header = hex HMAC-SHA256(body, secret)
  - Missing / empty LINEAR_WEBHOOK_SECRET → fail closed (403)
  - Self-loop guard: drops events where actor is the bot itself
  - Mention filter: only dispatches when body contains @escher-hermes

Inbound flow:
  POST /webhooks/linear-comments
    └─ _handle_linear_webhook()
         ├─ HMAC-SHA256 validation
         ├─ self-loop guard (actor.name / actor.email)
         ├─ mention filter (@escher-hermes regex)
         └─ handle_message() → per-issue agent session (thread key = issueId)

Outbound (send_message):
  POST https://api.linear.app/graphql
  Mutation: commentCreate(input:{issueId, body})
  Header: Authorization: <LINEAR_API_KEY>  (no Bearer prefix)
  After reply: posts a one-line summary to Slack #escher-bulletins

References:
  ESC-286 — Linear↔Hermes gateway platform adapter
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Linear GraphQL endpoint.
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

#: Signature header sent by Linear on every webhook POST.
LINEAR_SIG_HEADER = "Linear-Signature"

#: The bot's own Linear identity — used for self-loop guard.
_BOT_NAME = "escher-hermes"
_BOT_EMAIL = "escher-hermes+linear@imgix.com"

#: Mention filter: fire only when the comment body contains @escher-hermes
#: NOT followed by a word char or dash (avoids @escher-hermes-fake etc.).
_MENTION_RE = re.compile(r"(?i)@escher-hermes(?![\w-])")

#: Slack channel for awareness summaries after bot replies.
_BULLETINS_CHANNEL = "C0B4FJWFBC3"

# ---------------------------------------------------------------------------
# Lazy aiohttp guard
# ---------------------------------------------------------------------------

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Gateway imports (deferred so plugin discovery works in minimal env)
# ---------------------------------------------------------------------------

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Return True when all runtime dependencies are present."""
    return AIOHTTP_AVAILABLE


def _validate_config(config: PlatformConfig) -> bool:
    """Config is valid when LINEAR_API_KEY is set (env or extra)."""
    return bool(
        os.getenv("LINEAR_API_KEY")
        or config.extra.get("api_key")
    )


def _is_connected(config: PlatformConfig) -> bool:
    return _validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so gateway status reflects env-only config."""
    api_key = os.getenv("LINEAR_API_KEY", "")
    secret = os.getenv("LINEAR_WEBHOOK_SECRET", "")
    if not api_key:
        return None
    result: dict = {"api_key": api_key}
    if secret:
        result["webhook_secret"] = secret
    home = os.getenv("LINEAR_HOME_CHANNEL", "")
    if home:
        result["home_channel"] = home
    return result


# ---------------------------------------------------------------------------
# HMAC validation
# ---------------------------------------------------------------------------

def _validate_linear_signature(raw_body: bytes, secret: str, sig_header: str) -> bool:
    """Return True when ``sig_header`` matches hex HMAC-SHA256(body, secret).

    Linear sends: ``Linear-Signature: <lowercase hex>``

    Args:
        raw_body:   Raw request body bytes (must NOT be decoded/re-encoded).
        secret:     The webhook signing secret (LINEAR_WEBHOOK_SECRET).
        sig_header: Value of the ``Linear-Signature`` header.
    """
    if not secret or not sig_header:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(sig_header.strip(), expected)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Outbound: Linear GraphQL
# ---------------------------------------------------------------------------

async def _post_linear_comment(
    issue_id: str,
    body: str,
    api_key: str,
) -> dict:
    """Post a comment to a Linear issue via GraphQL.

    Returns a dict with ``success``, optionally ``comment_id``, ``comment_url``,
    or ``error``.
    """
    try:
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp not installed"}

    mutation = """
    mutation CreateComment($issueId: String!, $body: String!) {
      commentCreate(input: {issueId: $issueId, body: $body}) {
        success
        comment {
          id
          url
        }
      }
    }
    """
    payload = {
        "query": mutation,
        "variables": {"issueId": issue_id, "body": body},
    }
    headers = {
        "Authorization": api_key,  # Linear uses raw key, NO "Bearer" prefix
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(
                LINEAR_GRAPHQL_URL,
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {text[:200]}",
                    }
                data = await resp.json()
    except Exception as exc:
        return {"success": False, "error": f"Request failed: {exc}"}

    # GraphQL errors on HTTP 200
    if data.get("errors"):
        errs = "; ".join(e.get("message", str(e)) for e in data["errors"])
        return {"success": False, "error": f"GraphQL errors: {errs}"}

    result = data.get("data", {}).get("commentCreate", {})
    if not result.get("success"):
        return {"success": False, "error": "commentCreate returned success=false"}

    comment = result.get("comment") or {}
    return {
        "success": True,
        "comment_id": comment.get("id"),
        "comment_url": comment.get("url"),
    }


# ---------------------------------------------------------------------------
# Outbound: Slack awareness summary
# ---------------------------------------------------------------------------

async def _post_slack_bulletin(message: str) -> None:
    """Fire-and-forget: post a one-liner to #escher-bulletins.

    Reuses the same _send_slack helper as the rest of the gateway.
    Errors are logged but never raised — the bulletin is best-effort.
    """
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        logger.debug("[linear] SLACK_BOT_TOKEN not set — skipping bulletin")
        return
    try:
        from tools.send_message_tool import _send_slack  # type: ignore[attr-defined]
        result = await _send_slack(token, _BULLETINS_CHANNEL, message)
        if not result.get("success"):
            logger.warning("[linear] Slack bulletin failed: %s", result.get("error"))
    except Exception as exc:
        logger.warning("[linear] Slack bulletin error: %s", exc)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LinearAdapter(BasePlatformAdapter):
    """Linear gateway adapter — inbound @mention replies + outbound comments.

    Does NOT run a standalone HTTP server.  Instead it registers
    ``_handle_linear_webhook`` into the webhook adapter's shared :8644 aiohttp
    app via ``gateway.platforms.webhook._plugin_route_registry``.
    """

    # Maximum Linear comment length (API limit is ~65 000 chars; keep it
    # comfortable for humans reading the issue).
    MAX_MESSAGE_LENGTH = 10_000

    def __init__(self, config: PlatformConfig):
        # Platform._missing_() dynamically creates a pseudo-member for unknown
        # plugin platform names, so Platform("linear") is identity-stable and
        # works with the rest of the gateway machinery.
        from gateway.config import Platform as _Platform
        super().__init__(config, _Platform("linear"))

        self._api_key: str = (
            os.getenv("LINEAR_API_KEY", "") or config.extra.get("api_key", "")
        )
        self._webhook_secret: str = (
            os.getenv("LINEAR_WEBHOOK_SECRET", "")
            or config.extra.get("webhook_secret", "")
        )
        self._home_channel: str = (
            os.getenv("LINEAR_HOME_CHANNEL", "")
            or config.extra.get("home_channel", "")
        )

        # Cache: issue_id → {identifier, title} for Slack bulletin enrichment.
        # Populated on inbound webhook events; keyed by Linear UUID.
        self._issue_cache: Dict[str, dict] = {}

        # gateway_runner is injected by GatewayRunner._create_adapter()
        self.gateway_runner: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self._api_key:
            logger.error(
                "[linear] LINEAR_API_KEY is not set — cannot connect"
            )
            return False
        if not self._webhook_secret:
            logger.warning(
                "[linear] LINEAR_WEBHOOK_SECRET is not set — "
                "inbound webhooks will be rejected (fail-closed)"
            )
            # Still return True: outbound send_message works without a secret.
            # Inbound handler returns 403 on every request when secret is empty.

        # Register our handler in the module-level plugin route registry so
        # the webhook adapter mounts it on /webhooks/linear-comments when it
        # starts.  If the webhook adapter already started (unusual), the handler
        # won't be mounted — log a warning.
        from gateway.platforms.webhook import _plugin_route_registry
        _plugin_route_registry["/webhooks/linear-comments"] = (
            self._handle_linear_webhook
        )
        logger.info(
            "[linear] Registered /webhooks/linear-comments handler "
            "(will be mounted when webhook adapter starts)"
        )

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        # Remove our entry from the registry so a reconnect cycle doesn't
        # accumulate stale handlers.
        try:
            from gateway.platforms.webhook import _plugin_route_registry
            _plugin_route_registry.pop("/webhooks/linear-comments", None)
        except Exception:
            pass
        self._mark_disconnected()
        logger.info("[linear] Disconnected")

    # ------------------------------------------------------------------
    # HTTP handler (inbound)
    # ------------------------------------------------------------------

    async def _handle_linear_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/linear-comments handler.

        1. Validate Linear-Signature HMAC-SHA256
        2. Parse JSON body
        3. Self-loop guard (drop bot's own events)
        4. Mention filter (only @escher-hermes comments proceed)
        5. Dispatch handle_message → per-issue agent session
        """
        # Read body first (needed for HMAC)
        try:
            raw_body = await request.read()
        except Exception as exc:
            logger.error("[linear] Failed to read body: %s", exc)
            return web.json_response({"error": "Bad request"}, status=400)

        # Validate HMAC-SHA256 signature
        sig = request.headers.get(LINEAR_SIG_HEADER, "")
        if not self._webhook_secret:
            logger.error(
                "[linear] LINEAR_WEBHOOK_SECRET not set — rejecting request (fail-closed)"
            )
            return web.json_response(
                {"error": "Webhook secret not configured"}, status=403
            )
        if not _validate_linear_signature(raw_body, self._webhook_secret, sig):
            logger.warning("[linear] Invalid Linear-Signature — rejecting")
            return web.json_response({"error": "Invalid signature"}, status=401)

        # Parse JSON
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            logger.warning("[linear] Bad JSON body: %s", exc)
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Only handle Comment events
        event_type = payload.get("type", "")
        if event_type != "Comment":
            logger.debug("[linear] Ignoring event type: %s", event_type)
            return web.json_response(
                {"status": "ignored", "reason": "event_type", "type": event_type}
            )

        data = payload.get("data", {})
        actor = payload.get("actor", {})

        # Self-loop guard — drop events originated by the bot itself
        actor_name = (actor.get("name") or "").strip()
        actor_email = (actor.get("email") or "").strip()
        if actor_name == _BOT_NAME or actor_email == _BOT_EMAIL:
            logger.debug(
                "[linear] Self-loop guard: dropping event from bot actor (%s / %s)",
                actor_name,
                actor_email,
            )
            return web.json_response({"status": "ignored", "reason": "self_loop"})

        # Mention filter — only wake agent when @escher-hermes appears
        comment_body = data.get("body", "")
        if not _MENTION_RE.search(comment_body):
            logger.debug("[linear] No @escher-hermes mention — ignoring")
            return web.json_response({"status": "ignored", "reason": "no_mention"})

        # Extract issue context
        issue_id: str = data.get("issueId", "")
        issue_title: str = (data.get("issue") or {}).get("title", "")
        issue_identifier: str = (data.get("issue") or {}).get("identifier", issue_id)
        comment_id: str = data.get("id", "")
        commenter_name: str = actor_name or "Unknown"

        if not issue_id:
            logger.warning("[linear] Comment event missing issueId — ignoring")
            return web.json_response({"error": "Missing issueId"}, status=400)

        # Cache issue metadata for use in outbound Slack bulletins
        self._issue_cache[issue_id] = {
            "identifier": issue_identifier,
            "title": issue_title,
        }

        # Build session source — keyed per issue so follow-ups keep context
        session_chat_id = f"linear:{issue_id}"
        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"Linear/{issue_identifier}",
            chat_type="linear_issue",
            user_id=actor.get("id", commenter_name),
            user_name=commenter_name,
        )

        prompt = (
            f"[Linear issue {issue_identifier}: {issue_title}]\n\n"
            f"{commenter_name} said:\n{comment_body}"
        )

        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=comment_id,
        )

        logger.info(
            "[linear] Dispatching agent for issue %s (comment %s, actor %s)",
            issue_identifier,
            comment_id,
            commenter_name,
        )

        # Non-blocking dispatch — handle_message() manages background tasks
        # internally; the agent reply flows back through send() automatically.
        asyncio.create_task(self.handle_message(event))

        return web.json_response(
            {
                "status": "accepted",
                "issue": issue_identifier,
                "comment": comment_id,
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Post *content* as a Linear comment.

        chat_id is ``linear:<issueId>`` (set in _handle_linear_webhook).
        """
        # Extract issue_id from chat_id
        if chat_id.startswith("linear:"):
            issue_id = chat_id[len("linear:"):]
        else:
            # Fallback: treat chat_id as a raw issue ID
            issue_id = chat_id

        if not issue_id:
            logger.error("[linear] send() called with empty issue_id (chat_id=%r)", chat_id)
            return SendResult(success=False, error="Empty issue_id")

        if not self._api_key:
            return SendResult(success=False, error="LINEAR_API_KEY not set")

        result = await _post_linear_comment(issue_id, content, self._api_key)

        if result["success"]:
            logger.info(
                "[linear] Posted comment to issue %s: %s",
                issue_id,
                result.get("comment_url", ""),
            )
            # Build a human-readable Slack bulletin line.
            # Format: "ESC-XXX <title>: <what was done>."
            cached = self._issue_cache.get(issue_id, {})
            identifier = cached.get("identifier", issue_id)
            title = cached.get("title", "")
            # Truncate agent reply to a one-liner for the bulletin
            summary = content.split("\n")[0][:120].rstrip()
            if title:
                bulletin = f"{identifier} {title}: {summary}"
            else:
                bulletin = f"{identifier}: {summary}"
            await _post_slack_bulletin(bulletin)
        else:
            logger.error("[linear] commentCreate failed: %s", result.get("error"))

        return SendResult(
            success=result["success"],
            message_id=result.get("comment_id"),
            error=result.get("error"),
        )

    async def send_typing(self, chat_id: str) -> None:
        """Linear has no typing indicator API — no-op."""

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str = "",
    ) -> SendResult:
        """Deliver image as a markdown link in a Linear comment."""
        body = f"![image]({image_url})"
        if caption:
            body = f"{caption}\n\n{body}"
        return await self.send(chat_id, body)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "linear_issue", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Stand-alone sender (for cron out-of-process delivery)
# ---------------------------------------------------------------------------

async def _standalone_send(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> dict:
    """Out-of-process cron delivery: post a comment to Linear directly."""
    api_key = os.getenv("LINEAR_API_KEY", "") or getattr(pconfig, "extra", {}).get("api_key", "")
    if not api_key:
        return {"error": "LINEAR_API_KEY not set"}

    # chat_id may be "linear:<issueId>" or a bare issue ID
    if chat_id.startswith("linear:"):
        issue_id = chat_id[len("linear:"):]
    else:
        issue_id = chat_id

    return await _post_linear_comment(issue_id, message, api_key)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system on startup.

    Registers the Linear platform adapter.  The adapter:
    - Piggybacks on the webhook adapter's shared :8644 aiohttp app via the
      module-level ``_plugin_route_registry`` (see gateway/platforms/webhook.py)
    - Does NOT run a second HTTP server
    - Handles POST /webhooks/linear-comments with full HMAC validation
    """
    ctx.register_platform(
        name="linear",
        label="Linear",
        adapter_factory=lambda cfg: LinearAdapter(cfg),
        check_fn=check_requirements,
        validate_config=_validate_config,
        is_connected=_is_connected,
        required_env=["LINEAR_API_KEY", "LINEAR_WEBHOOK_SECRET"],
        install_hint="pip install aiohttp (already present with the webhook adapter)",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="LINEAR_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="LINEAR_ALLOWED_USERS",
        allow_all_env="LINEAR_ALLOW_ALL_USERS",
        max_message_length=LinearAdapter.MAX_MESSAGE_LENGTH,
        emoji="📋",
        platform_hint=(
            "You are responding in a Linear issue comment thread. "
            "Use concise Markdown — Linear renders headers, bold, code blocks, "
            "and bullet lists. Keep replies focused and actionable."
        ),
    )
    logger.info("[linear] Platform adapter registered")
