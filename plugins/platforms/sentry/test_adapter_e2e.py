"""End-to-end handler test for the Sentry adapter using a real aiohttp request.

POSTs synthetic signed payloads through _handle_sentry_webhook and asserts the
HTTP status codes for: valid signed alert (202), bad signature (401), missing
secret (403), bad JSON (400), non-actionable payload (200/ignored). The agent
dispatch (handle_message) is stubbed so no agent actually runs.
"""
import asyncio
import hashlib
import hmac
import json
import sys

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

# Real gateway imports work here (run from repo root with venv).
from plugins.platforms.sentry import adapter as A
from gateway.config import PlatformConfig

SECRET = "e2e-client-secret-xyz"

def sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def main() -> int:
    cfg = PlatformConfig(extra={"client_secret": SECRET, "alerts_channel": "#test"})
    ad = A.SentryAdapter(cfg)

    dispatched = []
    async def fake_handle(event):
        dispatched.append(event)
    ad.handle_message = fake_handle  # type: ignore[assignment]

    passed = failed = 0
    def check(name, cond):
        nonlocal passed, failed
        if cond: passed += 1; print(f"  PASS  {name}")
        else: failed += 1; print(f"  FAIL  {name}")

    async def post(body: bytes, sig: str | None, resource: str = "issue"):
        headers = {"Content-Type": "application/json"}
        if sig is not None:
            headers[A.SENTRY_SIG_HEADER] = sig
        headers[A.SENTRY_RESOURCE_HEADER] = resource
        req = make_mocked_request("POST", "/webhooks/sentry", headers=headers, payload=None)
        # make_mocked_request doesn't carry a body reader for our bytes; patch read.
        async def _read():
            return body
        req.read = _read  # type: ignore[assignment]
        return await ad._handle_sentry_webhook(req)

    issue_body = json.dumps({
        "action": "created",
        "data": {"issue": {
            "title": "ValueError in usage metering",
            "culprit": "/v1/usage",
            "level": "error",
            "project": {"slug": "escher-silvertip"},
            "permalink": "https://imgix.sentry.io/issues/1/",
        }},
    }).encode()

    print("== end-to-end handler ==")
    r = await post(issue_body, sign(issue_body, SECRET))
    check("valid signed alert -> 202", r.status == 202)
    # handler dispatches via asyncio.create_task; yield so the task runs.
    await asyncio.sleep(0)
    check("valid alert dispatched an agent event", len(dispatched) == 1)

    r = await post(issue_body, sign(issue_body, "nope"))
    check("bad signature -> 401", r.status == 401)

    r = await post(issue_body, "")
    check("missing signature -> 401", r.status == 401)

    r = await post(b"{not json", sign(b"{not json", SECRET))
    check("bad JSON -> 400", r.status == 400)

    noop_body = json.dumps({"action": "x", "data": {}}).encode()
    r = await post(noop_body, sign(noop_body, SECRET), resource="unknown")
    check("non-actionable -> 200 ignored", r.status == 200)

    # fail-closed: empty secret
    ad._client_secret = ""
    r = await post(issue_body, sign(issue_body, SECRET))
    check("no client secret -> 403 (fail-closed)", r.status == 403)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


sys.exit(asyncio.run(main()))
