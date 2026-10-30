"""Bounded opt-in results for trusted plugin consumers; transport is synthetic."""
import json
import pytest
from tests.hermes_cli.test_plugin_call_mcp import _make_ctx, _patch_config, _patch_handler
from hermes_cli.plugins import PluginContext


def test_large_json_complete_only_with_explicit_allowance(monkeypatch):
    _patch_config(monkeypatch, {'my-plugin': {'mcp_allowlist': ['fixture']}})
    payload={'account':{'id':'SYNTHETIC','notes':'x'*90000}}
    raw=json.dumps({'result':json.dumps(payload), 'structuredContent':payload})
    _patch_handler(monkeypatch,raw)
    ctx=_make_ctx()
    old=ctx.call_mcp('fixture','read_account')
    assert old['truncated'] is True
    large=ctx.call_mcp('fixture','read_account',max_result_chars=1_000_000)
    assert not large.get('truncated')
    assert json.loads(large['result'])==payload
    assert large['structuredContent']==payload
    assert large['response_chars']==len(raw) and large['result_limit_chars']==1_000_000
    assert ctx.call_mcp('fixture','read_account')['truncated'] is True
    assert PluginContext._MCP_RESULT_CHAR_CAP==65536


@pytest.mark.parametrize('limit',[0,-1,True,False,1.5,'1000000',2_000_001,float('inf')])
def test_bad_allowance_rejected_before_transport(monkeypatch,limit):
    _patch_config(monkeypatch, {'my-plugin': {'mcp_allowlist': ['fixture']}})
    called={};_patch_handler(monkeypatch,'{}',called)
    with pytest.raises(ValueError): _make_ctx().call_mcp('fixture','read',max_result_chars=limit)
    assert not called


def test_large_allowance_never_bypasses_server_permission(monkeypatch):
    _patch_config(monkeypatch,{})
    called={};_patch_handler(monkeypatch,'{}',called)
    with pytest.raises(PermissionError):_make_ctx().call_mcp('fixture','read',max_result_chars=1_000_000)
    assert not called


def test_new_bound_remains_fail_closed_for_oversize(monkeypatch):
    _patch_config(monkeypatch, {'my-plugin': {'mcp_allowlist': ['fixture']}})
    raw=json.dumps({'result':'x'*150000});_patch_handler(monkeypatch,raw)
    result=_make_ctx().call_mcp('fixture','read',max_result_chars=100000)
    assert result['truncated'] is True
    assert result['response_chars']==len(raw) and result['result_limit_chars']==100000


def test_exact_boundary_and_structured_errors_preserved(monkeypatch):
    _patch_config(monkeypatch, {'my-plugin': {'mcp_allowlist': ['fixture']}})
    raw=json.dumps({'error':'SYNTHETIC read denied'});_patch_handler(monkeypatch,raw)
    result=_make_ctx().call_mcp('fixture','read',max_result_chars=len(raw))
    assert result['ok'] is False and result['error']=='SYNTHETIC read denied'
    assert not result.get('truncated')
