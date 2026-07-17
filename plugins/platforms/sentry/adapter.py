"""
Sentry platform adapter for Hermes Agent.

Receives Sentry alert webhooks on /webhooks/sentry via the shared :8644 ingress
(gateway/platforms/webhook.py _plugin_route_registry). Does NOT run a second
HTTP server.

Architecture (see escher session 2026-06-03, "Sentry integration"):
  Sentry is a ONE-DIRECTIONAL alert source. Unlike the Linear adapter (which
  replies into the same comment thread), Sentry has no reply surface. The flow:

    Sentry alert rule fires
      └─ POST /webhooks/sentry
           ├─ HMAC-SHA256 verify (Sentry-Hook-Signature vs client secret, raw body)
           ├─ parse Internal-Integration payload (action + data.issue / data.event)
           ├─ resource filter (issue / event_alert / metric_alert only)
           ├─ build a triage prompt (project, title, culprit, count, level,
           │    release, permalink) + ESCHER project→service map
           └─ handle_message() → ONE-SHOT agent session

  The per-alert agent run does the real work via ITS OWN tools: judge severity,
  pull extra context with the read-only Sentry token, post to Slack, and (once
  out of observe-only) file an ESC ticket. The adapter's send() delivers the
  agent's final triage write-up to Slack #escher-alerts — that channel IS the
  reply surface for a source that can't be replied to.

Escalation discipline (Chris, 2026-06-03):
  The intelligence that decides "is this worth a full agent run?" lives in the
  Sentry ALERT RULES (which the agent curates over time), NOT in this handler.
  By the time an alert reaches /webhooks/sentry it is, by construction, focused
  enough to warrant a run. So this handler does NOT re-filter on severity — it
  verifies, parses, and dispatches. Tune at the rule layer, not here.

Security:
  - Validates ``Sentry-Hook-Signature: <hex>`` = hex HMAC-SHA256(raw_body, secret)
    using the integration Client Secret. Verify against RAW body BEFORE json
    parse (re-serialising changes bytes and breaks the signature — Sentry docs).
  - Missing / empty SENTRY_CLIENT_SECRET → fail closed (403).

References:
  Sentry integration platform webhooks:
    https://docs.sentry.io/integrations/integration-platform/webhooks/
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Signature header Sentry sends on every webhook POST.
SENTRY_SIG_HEADER = "Sentry-Hook-Signature"

#: Resource header Sentry sends identifying the payload kind.
SENTRY_RESOURCE_HEADER = "Sentry-Hook-Resource"

#: Default Slack channel for alert triage (override: SENTRY_ALERTS_CHANNEL).
#: NOTE: placeholder until #escher-alerts is created + its ID wired in env.
_DEFAULT_ALERTS_CHANNEL = "#escher-alerts"

#: Default Sentry org slug.
_DEFAULT_ORG_SLUG = "imgix"

#: Fallback path for the read-only auth token (out-of-band convention).
_TOKEN_FILE = Path.home() / ".hermes" / "secrets" / "sentry-token"

#: Map Sentry project slug -> the Escher service it represents, so the triage
#: prompt tells the agent which repo/runtime to reason about. The classic-imgix
#: projects (web-dashboard, blacktip) are intentionally NOT Escher's; left out
#: so an alert from them is clearly flagged as "not an Escher service".
_PROJECT_SERVICE_MAP: Dict[str, str] = {
    "escher-silvertip": "silvertip-api (central FastAPI server)",
    "escher-rust": "escher-rust (node-graph processing engine)",
    "escher-workers": "escher-workers (Pub/Sub job processors)",
    "escher-prism": "prism-ui (web dashboard / graph editor)",
}

# ---------------------------------------------------------------------------
# Lazy aiohttp guard
# ---------------------------------------------------------------------------

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

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


def _read_token_file() -> str:
    """Read the read-only Sentry auth token from the out-of-band secret file.

    Strips trailing whitespace/newline — the file is stored with a trailing
    \\n (confirmed 2026-06-03) and a byte-exact mismatch would break API auth.
    """
    try:
        return _TOKEN_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _resolve_client_secret(config: PlatformConfig) -> str:
    return (
        os.getenv("SENTRY_CLIENT_SECRET", "")
        or config.extra.get("client_secret", "")
    ).strip()


def _validate_config(config: PlatformConfig) -> bool:
    """Config is valid when the client secret is set (env or extra).

    The client secret is what secures the inbound webhook; without it the
    handler fails closed, so it is the minimum bar for "configured".
    """
    return bool(_resolve_client_secret(config))


def _is_connected(config: PlatformConfig) -> bool:
    return _validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so gateway status reflects env config."""
    secret = os.getenv("SENTRY_CLIENT_SECRET", "")
    if not secret:
        return None
    result: dict = {"client_secret": secret}
    token = os.getenv("SENTRY_AUTH_TOKEN", "")
    if token:
        result["auth_token"] = token
    channel = os.getenv("SENTRY_ALERTS_CHANNEL", "")
    if channel:
        result["alerts_channel"] = channel
    org = os.getenv("SENTRY_ORG_SLUG", "")
    if org:
        result["org_slug"] = org
    return result


# ---------------------------------------------------------------------------
# HMAC validation
# ---------------------------------------------------------------------------

def _validate_sentry_signature(raw_body: bytes, secret: str, sig_header: str) -> bool:
    """Return True when ``sig_header`` == hex HMAC-SHA256(raw_body, secret).

    Sentry signs with the integration Client Secret and sends the hex digest in
    the ``Sentry-Hook-Signature`` header. MUST be computed over the RAW request
    body before any JSON parse/re-serialise (per Sentry docs).
    """
    if not secret or not sig_header:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(sig_header.strip(), expected)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def _extract_alert(payload: dict, resource: str) -> Optional[dict]:
    """Normalise a Sentry webhook payload into a flat triage dict.

    Handles the Internal-Integration shapes:
      * resource "issue"        → data.issue           (issue lifecycle alert)
      * resource "event_alert"  → data.event + data.   (issue alert rule action)
      * resource "metric_alert" → data.metric_alert / data.description

    Returns a dict of the fields the triage prompt needs, or None when the
    payload carries nothing actionable (caller logs + 202s so Sentry doesn't
    retry).
    """
    data = payload.get("data") or {}
    action = payload.get("action", "")

    # Issue-alert rule action: richest shape, carries the event + triggered rule.
    if resource == "event_alert" or ("event" in data):
        event = data.get("event") or {}
        issue_url = event.get("issue_url") or event.get("web_url") or ""
        title = (
            event.get("title")
            or event.get("message")
            or (event.get("metadata") or {}).get("value")
            or "(untitled event)"
        )
        return {
            "kind": "event_alert",
            "action": action,
            "title": title,
            "culprit": event.get("culprit") or event.get("transaction") or "",
            "level": event.get("level") or "error",
            "environment": event.get("environment") or "",
            "release": event.get("release") or "",
            "project_slug": event.get("project") or payload.get("project") or "",
            "permalink": issue_url or event.get("url") or "",
            "rule": (data.get("triggered_rule") or ""),
        }

    # Issue lifecycle (created / resolved / regression / assigned / ignored).
    if resource == "issue" or ("issue" in data):
        issue = data.get("issue") or {}
        meta = issue.get("metadata") or {}
        return {
            "kind": "issue",
            "action": action,
            "title": issue.get("title") or meta.get("value") or "(untitled issue)",
            "culprit": issue.get("culprit") or "",
            "level": issue.get("level") or (meta.get("type") or "error"),
            "environment": "",
            "release": (issue.get("firstRelease") or {}).get("version", "")
            if isinstance(issue.get("firstRelease"), dict) else "",
            "project_slug": (issue.get("project") or {}).get("slug", "")
            if isinstance(issue.get("project"), dict) else "",
            "permalink": issue.get("permalink") or issue.get("web_url") or "",
            "count": issue.get("count") or "",
            "user_count": issue.get("userCount") or "",
        }

    # Metric alert (error-rate / performance threshold).
    if resource == "metric_alert" or ("metric_alert" in data):
        ma = data.get("metric_alert") or {}
        rule = ma.get("alert_rule") or {}
        return {
            "kind": "metric_alert",
            "action": action,
            "title": rule.get("name") or "(metric alert)",
            "culprit": "",
            "level": ma.get("status") or "critical",
            "environment": "",
            "release": "",
            "project_slug": "",
            "permalink": data.get("description_text") or "",
        }

    return None


def _build_triage_prompt(alert: dict, org_slug: str) -> str:
    """Assemble the one-shot agent prompt from a normalised alert dict."""
    slug = alert.get("project_slug") or "?"
    service = _PROJECT_SERVICE_MAP.get(slug)
    if service:
        svc_line = f"Service: {service}  (Sentry project `{slug}`)"
    elif slug in {"web-dashboard", "blacktip"}:
        svc_line = (
            f"Sentry project `{slug}` — this is a CLASSIC-IMGIX project, NOT an "
            f"Escher service. Triage accordingly (likely belongs to another team)."
        )
    else:
        svc_line = f"Sentry project `{slug}` (unmapped — identify what this is)"

    lines = [
        f"[Sentry alert — {alert.get('kind')} / action={alert.get('action') or '?'}]",
        svc_line,
        f"Title: {alert.get('title')}",
    ]
    if alert.get("culprit"):
        lines.append(f"Culprit: {alert['culprit']}")
    if alert.get("level"):
        lines.append(f"Level: {alert['level']}")
    if alert.get("environment"):
        lines.append(f"Environment: {alert['environment']}")
    if alert.get("release"):
        lines.append(f"Release: {alert['release']}")
    if alert.get("count"):
        lines.append(f"Event count: {alert['count']}  (users: {alert.get('user_count') or '?'})")
    if alert.get("rule"):
        lines.append(f"Triggered rule: {alert['rule']}")
    if alert.get("permalink"):
        lines.append(f"Sentry link: {alert['permalink']}")

    lines.append(
        "\nYou are triaging a live Sentry alert. Assess severity and likely "
        "cause, correlate to a recent deploy/release if relevant, and write a "
        "concise triage for the team. OBSERVE-ONLY for now: post your triage; "
        "do NOT file a Linear ticket yet unless explicitly enabled. Your final "
        "message is delivered to the Slack alerts channel verbatim."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outbound: Slack alert post
# ---------------------------------------------------------------------------

async def _post_slack(channel: str, message: str) -> dict:
    """Post *message* to a Slack channel via the shared gateway helper."""
    try:
        # Route through the generic registry shim (#41112): upstream moved
        # _send_slack into the slack plugin; the shim survives relocations.
        from tools.send_message_tool import _registry_standalone_send
        return await _registry_standalone_send("slack", None, channel, message)
    except Exception as exc:
        logger.warning("[sentry] Slack post error: %s", exc)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SentryAdapter(BasePlatformAdapter):
    """Sentry gateway adapter — inbound alert webhook → one-shot triage session.

    Registers ``_handle_sentry_webhook`` into the webhook adapter's shared
    :8644 aiohttp app via ``gateway.platforms.webhook._plugin_route_registry``.
    Does NOT run a standalone HTTP server. ``send()`` posts the agent's triage
    to Slack #escher-alerts (Sentry has no native reply surface).
    """

    MAX_MESSAGE_LENGTH = 8_000

    def __init__(self, config: PlatformConfig):
        from gateway.config import Platform as _Platform
        super().__init__(config, _Platform("sentry"))

        self._client_secret: str = _resolve_client_secret(config)
        self._auth_token: str = (
            os.getenv("SENTRY_AUTH_TOKEN", "")
            or config.extra.get("auth_token", "")
            or _read_token_file()
        ).strip()
        self._alerts_channel: str = (
            os.getenv("SENTRY_ALERTS_CHANNEL", "")
            or config.extra.get("alerts_channel", "")
            or _DEFAULT_ALERTS_CHANNEL
        )
        self._org_slug: str = (
            os.getenv("SENTRY_ORG_SLUG", "")
            or config.extra.get("org_slug", "")
            or _DEFAULT_ORG_SLUG
        )
        self.gateway_runner: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self._client_secret:
            logger.warning(
                "[sentry] SENTRY_CLIENT_SECRET not set — inbound alerts will be "
                "rejected (fail-closed). Adapter still registers the route."
            )

        from gateway.platforms.webhook import _plugin_route_registry
        _plugin_route_registry["/webhooks/sentry"] = self._handle_sentry_webhook
        logger.info(
            "[sentry] Registered /webhooks/sentry handler "
            "(mounted when webhook adapter starts). Alerts channel=%s org=%s",
            self._alerts_channel, self._org_slug,
        )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        try:
            from gateway.platforms.webhook import _plugin_route_registry
            _plugin_route_registry.pop("/webhooks/sentry", None)
        except Exception:
            pass
        self._mark_disconnected()
        logger.info("[sentry] Disconnected")

    # ------------------------------------------------------------------
    # HTTP handler (inbound)
    # ------------------------------------------------------------------

    async def _handle_sentry_webhook(self, request: "web.Request") -> "web.Response":
        """POST /webhooks/sentry — verify, parse, dispatch one-shot triage.

        1. Read raw body (needed for HMAC — before any parse).
        2. Verify Sentry-Hook-Signature HMAC-SHA256 against client secret.
        3. Parse JSON; normalise via _extract_alert (by Sentry-Hook-Resource).
        4. Dispatch handle_message → one-shot triage agent session.
        """
        try:
            raw_body = await request.read()
        except Exception as exc:
            logger.error("[sentry] Failed to read body: %s", exc)
            return web.json_response({"error": "Bad request"}, status=400)

        # Fail-closed if no secret configured.
        if not self._client_secret:
            logger.error(
                "[sentry] SENTRY_CLIENT_SECRET not set — rejecting (fail-closed)"
            )
            return web.json_response(
                {"error": "Webhook secret not configured"}, status=403
            )

        sig = request.headers.get(SENTRY_SIG_HEADER, "")
        # TEMP (Sentry HMAC mismatch diag, 2026-06-04): capture ground truth on
        # the next rejected request so we can see exactly where the digest
        # diverges. REMOVE once signature verification is confirmed working.
        if not _validate_sentry_signature(raw_body, self._client_secret, sig):
            try:
                _computed = hmac.new(
                    self._client_secret.encode("utf-8"), raw_body, hashlib.sha256
                ).hexdigest()
                _all_sig_headers = {
                    k: v for k, v in request.headers.items()
                    if "sign" in k.lower() or "hook" in k.lower()
                }
                logger.warning(
                    "[sentry][DIAG] HMAC mismatch. body_len=%d secret_len=%d "
                    "recv_sig=%r computed=%r resource=%r content_type=%r "
                    "hook_headers=%r body_head=%r",
                    len(raw_body), len(self._client_secret),
                    sig, _computed,
                    request.headers.get(SENTRY_RESOURCE_HEADER, ""),
                    request.headers.get("Content-Type", ""),
                    _all_sig_headers,
                    raw_body[:120],
                )
            except Exception as _diag_exc:
                logger.warning("[sentry][DIAG] diag logging failed: %s", _diag_exc)
            logger.warning("[sentry] Invalid Sentry-Hook-Signature — rejecting")
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            logger.warning("[sentry] Bad JSON body: %s", exc)
            return web.json_response({"error": "Invalid JSON"}, status=400)

        resource = request.headers.get(SENTRY_RESOURCE_HEADER, "") or payload.get("resource", "")

        # Verification ping (Sentry sends an empty/installation event on setup).
        if resource == "installation" or payload.get("action") == "created" and "installation" in payload:
            logger.info("[sentry] Installation/verification webhook received — ack")
            return web.json_response({"status": "ok", "reason": "installation"})

        alert = _extract_alert(payload, resource)
        if not alert:
            logger.info(
                "[sentry] No actionable alert in payload (resource=%r action=%r) — ack",
                resource, payload.get("action"),
            )
            return web.json_response(
                {"status": "ignored", "reason": "no_actionable_alert"}
            )

        # Session keyed per project+title so repeated fires of the same issue
        # land in the same session (continuity) rather than spawning a fresh
        # run each time. Hash keeps the key bounded + safe.
        slug = alert.get("project_slug") or "unknown"
        title_key = hashlib.sha1(
            (alert.get("title") or "").encode("utf-8")
        ).hexdigest()[:12]
        session_chat_id = f"sentry:{slug}:{title_key}"

        source = self.build_source(
            chat_id=session_chat_id,
            chat_name=f"Sentry/{slug}",
            chat_type="sentry_alert",
            user_id="sentry-alert",
            user_name="Sentry",
        )

        prompt = _build_triage_prompt(alert, self._org_slug)

        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=title_key,
        )

        logger.info(
            "[sentry] Dispatching triage for %s alert on project=%s: %s",
            alert.get("kind"), slug, (alert.get("title") or "")[:80],
        )
        asyncio.create_task(self.handle_message(event))

        return web.json_response(
            {"status": "accepted", "project": slug, "kind": alert.get("kind")},
            status=202,
        )

    # ------------------------------------------------------------------
    # Outbound — the triage write-up goes to Slack #escher-alerts
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Deliver the agent's triage write-up to the Slack alerts channel.

        Sentry has no reply surface, so the per-alert agent run's final message
        is posted to #escher-alerts. The chat_id (``sentry:<slug>:<hash>``) is
        carried for logging/continuity but the destination is always the alerts
        channel. Any in-run Slack/Linear actions the agent took itself are
        independent of this delivery.
        """
        result = await _post_slack(self._alerts_channel, content)
        if result.get("success"):
            logger.info("[sentry] Triage posted to %s for %s", self._alerts_channel, chat_id)
        else:
            logger.error("[sentry] Triage post failed: %s", result.get("error"))
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("ts") or result.get("message_id"),
            error=result.get("error"),
        )

    async def send_typing(self, chat_id: str) -> None:
        """No typing indicator for an alert sink — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "sentry_alert", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system on startup.

    Registers the Sentry platform adapter. The adapter:
    - Piggybacks on the webhook adapter's shared :8644 aiohttp app via the
      module-level ``_plugin_route_registry`` (see gateway/platforms/webhook.py)
    - Does NOT run a second HTTP server
    - Handles POST /webhooks/sentry with HMAC validation against the client secret
    """
    ctx.register_platform(
        name="sentry",
        label="Sentry",
        adapter_factory=lambda cfg: SentryAdapter(cfg),
        check_fn=check_requirements,
        validate_config=_validate_config,
        is_connected=_is_connected,
        required_env=["SENTRY_CLIENT_SECRET"],
        install_hint="pip install aiohttp (already present with the webhook adapter)",
        env_enablement_fn=_env_enablement,
        allowed_users_env="SENTRY_ALLOWED_USERS",
        allow_all_env="SENTRY_ALLOW_ALL_USERS",
        max_message_length=SentryAdapter.MAX_MESSAGE_LENGTH,
        emoji="🚨",
        platform_hint=(
            "You are triaging a Sentry error alert. Be concise and actionable. "
            "Use Slack-friendly Markdown. Lead with severity and the one-line "
            "what-broke, then likely cause and suspect release, then a "
            "recommended next step. You have read-only Sentry API access and "
            "Linear tools available if you need more context."
        ),
    )
    logger.info("[sentry] Platform adapter registered")
