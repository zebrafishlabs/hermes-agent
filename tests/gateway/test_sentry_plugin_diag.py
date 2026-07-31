"""ESC-703: the Sentry webhook's signature-rejection path must never log
attacker-controlled or secret-derived material.

Background: a `TEMP (Sentry HMAC mismatch diag, 2026-06-04)` block shipped in
`plugins/platforms/sentry/adapter.py` and survived ~7 weeks past its own
"REMOVE once signature verification is confirmed working" note. On EVERY
failed-signature request it logged `secret_len`, the server-computed HMAC
digest over the caller's chosen body, all hook headers, and `raw_body[:120]`.

`/webhooks/sentry` is internet-reachable through the GCLB (Cloud Armor
default-allow on `/webhooks/*`, 1000 req/min/IP), and this branch runs on
UNAUTHENTICATED input — so that block was an unauthenticated path to
secret-length disclosure, a chosen-plaintext digest oracle, and log injection.

These tests are the regression guard. They assert the CONTRACT (the rejection
log carries no sensitive material), not the absence of one particular string,
so a differently-worded reintroduction of the same leak still fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging

from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_sentry = load_plugin_adapter("sentry")

SentryAdapter = _sentry.SentryAdapter
SENTRY_SIG_HEADER = _sentry.SENTRY_SIG_HEADER
SENTRY_RESOURCE_HEADER = _sentry.SENTRY_RESOURCE_HEADER

# Distinctive so a substring search cannot false-negative.
_SECRET = "esc703-sentry-client-secret-value"
_BODY = b'{"action":"triggered","canary":"ESC703-BODY-CANARY-abcdef"}'


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_request(headers=None, body=b""):
    req = MagicMock()
    req.headers = headers or {}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _make_adapter(secret=_SECRET):
    return SentryAdapter(
        PlatformConfig(enabled=True, extra={"client_secret": secret})
    )


def _expected_digest(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


class TestSignatureRejectionLogsNothingSensitive:
    """The rejection branch must stay content-free (ESC-703)."""

    def _reject_and_capture(self, caplog, headers=None, body=_BODY):
        adapter = _make_adapter()
        hdrs = {SENTRY_SIG_HEADER: "deadbeef" * 8}
        hdrs.update(headers or {})
        req = _mock_request(headers=hdrs, body=body)
        with caplog.at_level(logging.DEBUG):
            resp = _run(adapter._handle_sentry_webhook(req))
        return resp, "\n".join(r.getMessage() for r in caplog.records)

    def test_invalid_signature_still_rejects_with_401(self, caplog):
        """Security behavior unchanged: bad signature is refused."""
        resp, _ = self._reject_and_capture(caplog)
        assert resp.status == 401

    def test_rejection_log_omits_request_body(self, caplog):
        """Attacker-controlled bytes must never reach the log (log injection)."""
        _, logs = self._reject_and_capture(caplog)
        assert "ESC703-BODY-CANARY-abcdef" not in logs
        assert _BODY.decode() not in logs

    def test_rejection_log_omits_computed_digest(self, caplog):
        """Logging the computed HMAC turns the endpoint into a
        chosen-plaintext oracle: caller picks the body, reads back the
        digest the secret produced over it."""
        _, logs = self._reject_and_capture(caplog)
        assert _expected_digest(_BODY, _SECRET) not in logs

    def test_rejection_log_omits_secret_length(self, caplog):
        """`secret_len=N` narrows the search space for the shared secret."""
        _, logs = self._reject_and_capture(caplog)
        assert f"secret_len={len(_SECRET)}" not in logs
        assert "secret_len" not in logs

    def test_rejection_log_omits_secret_value(self, caplog):
        _, logs = self._reject_and_capture(caplog)
        assert _SECRET not in logs

    def test_rejection_log_omits_received_signature_and_headers(self, caplog):
        """Echoing caller headers back into the log is the same
        injection primitive as echoing the body."""
        _, logs = self._reject_and_capture(
            caplog,
            headers={
                SENTRY_RESOURCE_HEADER: "ESC703-RESOURCE-CANARY",
                "Sentry-Hook-Injected": "ESC703-HEADER-CANARY",
            },
        )
        assert "ESC703-HEADER-CANARY" not in logs
        assert "ESC703-RESOURCE-CANARY" not in logs
        assert "deadbeef" * 8 not in logs

    def test_no_diag_marker_remains(self, caplog):
        """The TEMP block tagged its lines [sentry][DIAG]; its return in any
        form should trip this."""
        _, logs = self._reject_and_capture(caplog)
        assert "DIAG" not in logs

    def test_rejection_is_still_observable(self, caplog):
        """Removing the leak must not make failures silent — an operator
        still needs to see that a rejection happened."""
        _, logs = self._reject_and_capture(caplog)
        assert "Invalid Sentry-Hook-Signature" in logs

    def test_missing_signature_header_also_rejects_quietly(self, caplog):
        """No-signature is the commonest scanner shape; same contract."""
        adapter = _make_adapter()
        req = _mock_request(headers={}, body=_BODY)
        with caplog.at_level(logging.DEBUG):
            resp = _run(adapter._handle_sentry_webhook(req))
        logs = "\n".join(r.getMessage() for r in caplog.records)
        assert resp.status == 401
        assert "ESC703-BODY-CANARY-abcdef" not in logs
        assert _expected_digest(_BODY, _SECRET) not in logs

    def test_unconfigured_secret_rejects_without_leaking(self, caplog):
        """Fail-closed path (no secret set) must also stay content-free."""
        adapter = _make_adapter(secret="")
        req = _mock_request(
            headers={SENTRY_SIG_HEADER: "abc"}, body=_BODY
        )
        with caplog.at_level(logging.DEBUG):
            resp = _run(adapter._handle_sentry_webhook(req))
        logs = "\n".join(r.getMessage() for r in caplog.records)
        assert resp.status == 403
        assert "ESC703-BODY-CANARY-abcdef" not in logs


class TestValidSignatureStillWorks:
    """The cold path must be unbroken by the removal."""

    def test_valid_signature_passes_the_gate(self, caplog):
        """A correctly-signed request must NOT be rejected at the signature
        check — it proceeds past 401/403 into normal handling."""
        adapter = _make_adapter()
        body = b'{"action":"triggered","data":{"issue":{"id":"1"}}}'
        sig = _expected_digest(body, _SECRET)
        req = _mock_request(
            headers={
                SENTRY_SIG_HEADER: sig,
                SENTRY_RESOURCE_HEADER: "issue",
            },
            body=body,
        )
        with caplog.at_level(logging.DEBUG):
            resp = _run(adapter._handle_sentry_webhook(req))
        logs = "\n".join(r.getMessage() for r in caplog.records)
        assert resp.status not in (401, 403)
        assert "Invalid Sentry-Hook-Signature" not in logs
