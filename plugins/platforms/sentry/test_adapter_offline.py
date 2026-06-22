"""Offline test for the Sentry adapter — no gateway, no network.

Exercises the security-critical and parsing logic against synthetic Sentry
payloads:
  1. HMAC verify accepts a correctly-signed body and rejects tampered/wrong-key
  2. _extract_alert normalises all three resource shapes + the empty case
  3. _build_triage_prompt maps escher projects + flags classic-imgix projects
"""
import hashlib
import hmac
import json
import sys

# Import the adapter module directly by path so we don't need the full gateway.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sentry_adapter",
    "plugins/platforms/sentry/adapter.py",
)
# Stub the gateway imports the adapter does at module load.
import types
for modname, attrs in {
    "gateway.platforms.base": ["BasePlatformAdapter", "MessageEvent", "MessageType", "SendResult"],
    "gateway.config": ["PlatformConfig", "Platform"],
}.items():
    m = types.ModuleType(modname)
    for a in attrs:
        setattr(m, a, type(a, (), {}))
    sys.modules.setdefault(modname.split(".")[0], types.ModuleType(modname.split(".")[0]))
    if "." in modname:
        sys.modules.setdefault(modname.rsplit(".", 1)[0], types.ModuleType(modname.rsplit(".", 1)[0]))
    sys.modules[modname] = m

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SECRET = "test-client-secret-abc123"
WRONG = "wrong-secret"

def sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}")

print("== 1. HMAC signature validation ==")
body = json.dumps({"action": "created", "data": {}}).encode()
good = sign(body, SECRET)
check("accepts correctly-signed body", mod._validate_sentry_signature(body, SECRET, good))
check("rejects wrong-key signature", not mod._validate_sentry_signature(body, WRONG, good))
check("rejects tampered body", not mod._validate_sentry_signature(body + b"x", SECRET, good))
check("rejects empty signature", not mod._validate_sentry_signature(body, SECRET, ""))
check("rejects empty secret (fail-closed)", not mod._validate_sentry_signature(body, "", good))

print("== 2. payload extraction ==")
# issue lifecycle (regression)
issue_payload = {
    "action": "created",
    "data": {"issue": {
        "title": "TypeError: Cannot read 'id' of undefined",
        "culprit": "/v1/graph-execute",
        "level": "error",
        "permalink": "https://imgix.sentry.io/issues/123/",
        "project": {"slug": "escher-silvertip"},
        "count": "42", "userCount": 7,
    }},
}
a = mod._extract_alert(issue_payload, "issue")
check("issue: kind", a and a["kind"] == "issue")
check("issue: title", a and "TypeError" in a["title"])
check("issue: project slug", a and a["project_slug"] == "escher-silvertip")
check("issue: count", a and a["count"] == "42")

# event_alert (issue alert rule action)
event_payload = {
    "action": "triggered",
    "data": {"event": {
        "title": "panic: index out of bounds",
        "transaction": "graph-execute",
        "level": "fatal",
        "environment": "production",
        "release": "abc123def",
        "project": "escher-rust",
        "issue_url": "https://imgix.sentry.io/issues/999/",
    }, "triggered_rule": "High volume errors"},
}
b = mod._extract_alert(event_payload, "event_alert")
check("event_alert: kind", b and b["kind"] == "event_alert")
check("event_alert: release", b and b["release"] == "abc123def")
check("event_alert: rule", b and b["rule"] == "High volume errors")
check("event_alert: env", b and b["environment"] == "production")

# empty / non-actionable
check("empty payload -> None", mod._extract_alert({"action": "x", "data": {}}, "unknown") is None)

print("== 3. triage prompt ==")
p1 = mod._build_triage_prompt(a, "imgix")
check("prompt maps escher service", "silvertip-api" in p1)
check("prompt includes title", "TypeError" in p1)
check("prompt includes observe-only directive", "OBSERVE-ONLY" in p1)
imgix_alert = dict(a); imgix_alert["project_slug"] = "web-dashboard"
p2 = mod._build_triage_prompt(imgix_alert, "imgix")
check("prompt flags classic-imgix as NOT escher", "NOT an" in p2)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
