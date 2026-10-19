"""Stable request identity must reach both observer hooks for dual-surface consent."""
from unittest.mock import patch

from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


def test_hooks_and_surface_share_exact_request_id():
    captured=[]; notified=[]; session='request-identity-test'
    def notify(data):
        notified.append(data)
        assert approval.resolve_gateway_approval(session,'once',request_id=data['request_id'])==1
    with patch('tools.approval_context._fire_approval_hook',side_effect=lambda name,**kw:captured.append((name,kw))):
        result=_await_gateway_decision(session,notify,{'command':'safe-test','description':'Test only','pattern_key':'test'})
    assert result['choice']=='once'
    assert len(notified)==1
    rid=notified[0]['request_id']
    assert rid
    assert [x[0] for x in captured]==['pre_approval_request','post_approval_response']
    assert all(data['request_id']==rid for _,data in captured)
    assert all(data['timeout_seconds']>0 for _,data in captured)
    assert approval.list_gateway_approvals(session)==[]


def test_same_command_in_different_sessions_has_different_ids():
    ids=[]
    def run(session):
        def notify(data):
            ids.append(data['request_id'])
            approval.resolve_gateway_approval(session,'deny',request_id=data['request_id'])
        return _await_gateway_decision(session,notify,{'command':'same-command','pattern_key':'test'})
    with patch('tools.approval_context._fire_approval_hook'):
        assert run('identity-a')['choice']=='deny'
        assert run('identity-b')['choice']=='deny'
    assert ids[0]!=ids[1]
