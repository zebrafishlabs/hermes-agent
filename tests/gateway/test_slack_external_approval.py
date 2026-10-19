"""ESC-415: Slack buttons are request-bound and external decisions retire them."""
import json
from unittest.mock import AsyncMock, patch

import pytest
from tests.gateway.test_slack_approval_buttons import _make_adapter, _attach_auth_runner
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


def click_body():
    return {'message':{'ts':'1.2','blocks':[{'type':'section','text':{'type':'mrkdwn','text':'Test command'}}]},
            'channel':{'id':'C1'},'user':{'name':'alice','id':'U_ALICE'}}


@pytest.mark.asyncio
async def test_button_carries_exact_request_identity():
    adapter=_make_adapter();client=adapter._team_clients['T1']
    client.chat_postMessage=AsyncMock(return_value={'ts':'1.2'})
    await adapter.send_exec_approval('C1','test-only','session',metadata={'approval_request_id':'request-a'})
    buttons=client.chat_postMessage.call_args.kwargs['blocks'][1]['elements']
    assert all(json.loads(x['value'])=={'session_key':'session','request_id':'request-a'} for x in buttons)


@pytest.mark.asyncio
async def test_stale_button_cannot_approve_new_request():
    adapter=_make_adapter();_attach_auth_runner(adapter);adapter._approval_resolved['1.2']=False
    session='stale-slack-button';new=_ApprovalEntry({'request_id':'new-request','command':'different-command'})
    with approval._lock:approval._gateway_queues[session]=[new]
    action={'action_id':'hermes_approve_once','value':json.dumps({'session_key':session,'request_id':'old-request'})}
    try:
        await adapter._handle_approval_action(AsyncMock(),click_body(),action)
        assert not new.event.is_set()
        assert approval.list_gateway_approvals(session)[0]['request_id']=='new-request'
    finally:
        approval.unregister_gateway_notify(session)


@pytest.mark.asyncio
async def test_session_only_legacy_button_fails_closed():
    adapter=_make_adapter();_attach_auth_runner(adapter);adapter._approval_resolved['1.2']=False
    action={'action_id':'hermes_approve_once','value':'session-without-request-id'}
    with patch('tools.approval.resolve_gateway_approval') as resolver:
        await adapter._handle_approval_action(AsyncMock(),click_body(),action)
    resolver.assert_not_called()


@pytest.mark.asyncio
async def test_valid_click_resolves_exact_request():
    adapter=_make_adapter();_attach_auth_runner(adapter);adapter._approval_resolved['1.2']=False
    action={'action_id':'hermes_approve_once','value':json.dumps({'session_key':'s','request_id':'r'})}
    with patch('tools.approval.resolve_gateway_approval',return_value=1) as resolver:
        await adapter._handle_approval_action(AsyncMock(),click_body(),action)
    resolver.assert_called_once_with('s','once',request_id='r',source='slack')


@pytest.mark.asyncio
async def test_external_retirement_removes_buttons_and_marks_resolved():
    adapter=_make_adapter();_attach_auth_runner(adapter);adapter._approval_resolved['1.2']=False
    await adapter.retire_exec_approval('C1','1.2','python3 test.py','Approved via Duo')
    kwargs=adapter._team_clients['T1'].chat_update.call_args.kwargs
    assert kwargs['ts']=='1.2'
    assert kwargs['text']=='Approved via Duo'
    assert not any(x['type']=='actions' for x in kwargs['blocks'])
    assert 'python3 test.py' in str(kwargs['blocks'])
    with patch('tools.approval.resolve_gateway_approval') as resolver:
        await adapter._handle_approval_action(AsyncMock(),click_body(),{'action_id':'hermes_approve_once','value':json.dumps({'session_key':'s','request_id':'r'})})
    resolver.assert_not_called()
