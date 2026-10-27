"""POST /v1/livekit/token (ESC-1159): bearer-gated LiveKit join-token mint for the
escher-chrome-plugin avatar pane. Credentials are monkeypatched — tests never touch
real secret files or the network."""

import base64
import hashlib
import hmac
import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter


API_KEY = "-".join(("fixture", "neutral", "api", "key", "123"))
LK_URL = "wss://fixture.livekit.cloud"
LK_KEY = "fixture-lk-key"
LK_SECRET = "fixture-lk-secret"


def _adapter(*, key=API_KEY):
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/livekit/token", adapter._handle_livekit_token)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _decode(token):
    segs = token.split(".")
    assert len(segs) == 3
    pad = lambda s: s + "=" * (-len(s) % 4)
    payload = json.loads(base64.urlsafe_b64decode(pad(segs[1])))
    expected = base64.urlsafe_b64encode(
        hmac.new(LK_SECRET.encode(), (segs[0] + "." + segs[1]).encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    assert segs[2] == expected, "JWT signature must verify against the fixture secret"
    return payload


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setattr(
        api_server_module, "_livekit_credentials", lambda: (LK_URL, LK_KEY, LK_SECRET))
    return None


@pytest.fixture
def no_creds(monkeypatch):
    monkeypatch.setattr(api_server_module, "_livekit_credentials", lambda: None)
    return None


async def _client(adapter):
    return TestClient(TestServer(_app(adapter)))


@pytest.mark.asyncio
async def test_requires_bearer(creds):
    async with await _client(_adapter()) as client:
        resp = await client.post("/v1/livekit/token", json={})
        assert resp.status == 401
        resp = await client.post(
            "/v1/livekit/token", json={}, headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401


@pytest.mark.asyncio
async def test_unconfigured_returns_503(no_creds):
    async with await _client(_adapter()) as client:
        resp = await client.post(
            "/v1/livekit/token", json={}, headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 503
        body = await resp.json()
        assert body["error"]["code"] == "livekit_not_configured"


@pytest.mark.asyncio
async def test_mints_dispatched_join_token(creds):
    async with await _client(_adapter()) as client:
        before = int(time.time())
        resp = await client.post(
            "/v1/livekit/token", json={},
            headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 200
        body = await resp.json()
        assert body["url"] == LK_URL
        assert body["room"].startswith("escher-avatar-")
        assert body["expires_in"] == 7200
        claims = _decode(body["token"])
        assert claims["iss"] == LK_KEY
        assert claims["sub"] == "Chris"  # documented default identity
        assert claims["exp"] - before >= 7200
        assert claims["video"] == {
            "room": body["room"], "roomJoin": True,
            "canPublish": True, "canSubscribe": True, "canPublishData": True}
        assert claims["roomConfig"]["agents"] == [{"agentName": "escher-avatar"}]
        assert claims["roomConfig"]["syncStreams"] is True


@pytest.mark.asyncio
async def test_rooms_are_unique_per_mint(creds):
    async with await _client(_adapter()) as client:
        rooms = set()
        for _ in range(3):
            resp = await client.post(
                "/v1/livekit/token", json={},
                headers={"Authorization": f"Bearer {API_KEY}"})
            rooms.add((await resp.json())["room"])
        assert len(rooms) == 3


@pytest.mark.asyncio
async def test_identity_and_ttl_overrides(creds):
    async with await _client(_adapter()) as client:
        resp = await client.post(
            "/v1/livekit/token", json={"identity": "  Tester  ", "ttl_seconds": 60},
            headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 200
        body = await resp.json()
        assert _decode(body["token"])["sub"] == "Tester"
        assert body["expires_in"] == 60
        # TTL clamps at the documented ceiling.
        resp = await client.post(
            "/v1/livekit/token", json={"ttl_seconds": 999999},
            headers={"Authorization": f"Bearer {API_KEY}"})
        assert (await resp.json())["expires_in"] == 14400


@pytest.mark.asyncio
async def test_rejects_bad_body(creds):
    async with await _client(_adapter()) as client:
        auth = {"Authorization": f"Bearer {API_KEY}"}
        resp = await client.post("/v1/livekit/token", json={"identity": ""}, headers=auth)
        assert resp.status == 400
        resp = await client.post("/v1/livekit/token", json={"identity": 42}, headers=auth)
        assert resp.status == 400
        resp = await client.post("/v1/livekit/token", json={"ttl_seconds": "7200"}, headers=auth)
        assert resp.status == 400
        resp = await client.post("/v1/livekit/token", json={"ttl_seconds": True}, headers=auth)
        assert resp.status == 400
        resp = await client.post(
            "/v1/livekit/token", data=b"not json",
            headers={**auth, "Content-Type": "application/json"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_capabilities_advertise_feature_and_endpoint(creds):
    async with await _client(_adapter()) as client:
        resp = await client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {API_KEY}"})
        body = await resp.json()
        assert body["features"]["livekit_token"] is True
        assert body["endpoints"]["livekit_token"] == {
            "method": "POST", "path": "/v1/livekit/token"}


@pytest.mark.asyncio
async def test_capabilities_hide_endpoint_when_unconfigured(no_creds):
    async with await _client(_adapter()) as client:
        resp = await client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {API_KEY}"})
        body = await resp.json()
        assert body["features"]["livekit_token"] is False
        assert "livekit_token" not in body["endpoints"]
