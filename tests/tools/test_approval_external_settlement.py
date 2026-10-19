"""External decisions must retire the exact card, even before its send returns."""
from unittest.mock import patch

from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


def test_settle_can_register_after_external_resolution():
    session='late-card-settlement'; seen=[]
    def notify(data):
        rid=data['request_id']
        assert approval.resolve_gateway_approval(session,'once',request_id=rid,source='duo')==1
        assert approval.register_gateway_settle(session,rid,
            lambda reason,**outcome:seen.append((reason,outcome)),with_outcome=True)
    with patch('tools.approval_context._fire_approval_hook'):
        result=_await_gateway_decision(session,notify,{'command':'test-only','pattern_key':'test'})
    assert result['choice']=='once'
    assert seen==[('resolved',{'choice':'once','source':'duo'})]
    assert not any(key[0]==session for key in approval._gateway_settle_entries)


def test_legacy_settle_callback_keeps_one_argument():
    session='legacy-card-callback';seen=[]
    def notify(data):
        assert approval.register_gateway_settle(session,data['request_id'],seen.append)
        approval.resolve_gateway_approval(session,'deny',request_id=data['request_id'],source='duo')
    with patch('tools.approval_context._fire_approval_hook'):
        assert _await_gateway_decision(session,notify,{'command':'test-only','pattern_key':'test'})['choice']=='deny'
    assert seen==['resolved']
