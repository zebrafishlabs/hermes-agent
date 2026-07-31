"""ESC-695: Linear and Sentry adapters must accept — and honor — the
``is_reconnect`` keyword on ``connect()``.

The gateway reconnect watcher calls ``adapter.connect(is_reconnect=True)``
on every retry tick. Both fork-local adapters (linear, sentry) were still
on the pre-kwarg signature ``connect(self)``, so every reconnect attempt
raised ``TypeError`` before any reconnect logic ran — the retry loop never
converged, and a dropped connection stayed dead until a manual gateway
restart (symptom: ``Reconnect linear error: ... unexpected keyword argument
'is_reconnect'`` every 300s in the gateway journal, forever).

Static signature conformance across ALL adapters is covered by
``test_adapter_connect_is_reconnect_contract.py``. These tests are the
behavioral proof for the two fixed adapters:

1. ``connect(is_reconnect=True)`` is accepted and returns a bool
   (direct reproduction of the TypeError — fails on pre-fix code).
2. Cold-start ``connect()`` (no kwarg) is unchanged.
3. Reconnect is idempotent: two sequential ``connect(is_reconnect=True)``
   calls leave exactly ONE route registration, not two (neither adapter
   opens sessions/tasks/sockets on connect — the route-registry dict write
   is the only side effect, and it must survive re-runs; cf. the resource
   audit precedent in ``test_platform_reconnect_fd_leak.py``).
"""

from __future__ import annotations

import asyncio

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_linear = load_plugin_adapter("linear")
_sentry = load_plugin_adapter("sentry")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_linear():
    return _linear.LinearAdapter(
        PlatformConfig(enabled=True, extra={"api_key": "k", "webhook_secret": "s"})
    )


def _make_sentry():
    return _sentry.SentryAdapter(
        PlatformConfig(enabled=True, extra={"client_secret": "s"})
    )


class TestLinearReconnect:
    ROUTE = "/webhooks/linear-comments"

    def _registry(self):
        from gateway.platforms.webhook import _plugin_route_registry
        return _plugin_route_registry

    def test_connect_accepts_is_reconnect_true(self):
        """Direct repro of the ESC-695 TypeError — must return a bool."""
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_linear()
        try:
            result = _run(adapter.connect(is_reconnect=True))
            assert isinstance(result, bool)
            assert result is True
        finally:
            reg.pop(self.ROUTE, None)

    def test_cold_start_connect_unchanged(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_linear()
        try:
            assert _run(adapter.connect()) is True
            assert self.ROUTE in reg
        finally:
            reg.pop(self.ROUTE, None)

    def test_reconnect_is_idempotent_single_registration(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_linear()
        try:
            assert _run(adapter.connect()) is True
            first_handler = reg[self.ROUTE]
            # Two reconnect ticks — registry must hold exactly one live
            # handler for the route, pointing at this adapter.
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert list(reg).count(self.ROUTE) == 1
            assert reg[self.ROUTE].__self__ is first_handler.__self__
        finally:
            reg.pop(self.ROUTE, None)

    def test_reconnect_after_disconnect_restores_route(self):
        """The real-world sequence: drop → reconnect must re-register."""
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_linear()
        try:
            assert _run(adapter.connect()) is True
            _run(adapter.disconnect())
            assert self.ROUTE not in reg
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert self.ROUTE in reg
        finally:
            reg.pop(self.ROUTE, None)


class TestSentryReconnect:
    ROUTE = "/webhooks/sentry"

    def _registry(self):
        from gateway.platforms.webhook import _plugin_route_registry
        return _plugin_route_registry

    def test_connect_accepts_is_reconnect_true(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_sentry()
        try:
            result = _run(adapter.connect(is_reconnect=True))
            assert isinstance(result, bool)
            assert result is True
        finally:
            reg.pop(self.ROUTE, None)

    def test_cold_start_connect_unchanged(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_sentry()
        try:
            assert _run(adapter.connect()) is True
            assert self.ROUTE in reg
        finally:
            reg.pop(self.ROUTE, None)

    def test_reconnect_is_idempotent_single_registration(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_sentry()
        try:
            assert _run(adapter.connect()) is True
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert list(reg).count(self.ROUTE) == 1
        finally:
            reg.pop(self.ROUTE, None)

    def test_reconnect_after_disconnect_restores_route(self):
        reg = self._registry()
        reg.pop(self.ROUTE, None)
        adapter = _make_sentry()
        try:
            assert _run(adapter.connect()) is True
            _run(adapter.disconnect())
            assert self.ROUTE not in reg
            assert _run(adapter.connect(is_reconnect=True)) is True
            assert self.ROUTE in reg
        finally:
            reg.pop(self.ROUTE, None)
