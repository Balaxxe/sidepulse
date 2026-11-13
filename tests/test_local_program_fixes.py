"""Custom hook behavior must reach the daemon's live normalized state."""
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from sidepulse.models import AgentMode
from sidepulse.program_adapter import hook_report
from sidepulse.program_presentation import ProgramMonitor
from sidepulse.program_status import ProgramState, StatusError


@pytest.fixture
def runtime():
    store = ProgramState()
    yield store, ProgramMonitor(store)
    store.close()


def report(event, provider='claude', **fields):
    return hook_report(provider, {'hook_event_name': event, 'session_id': 'local-parent', **fields})


@pytest.mark.parametrize('agent_fields', [{}, {'agent_id': 'child-agent'}])
def test_subagent_stop_clears_approval_in_new_accounting(runtime, agent_fields):
    store, monitor = runtime
    store.submit(report('PermissionRequest', tool_name='Bash', tool_use_id='pending', **agent_fields))
    assert monitor.snapshot().aggregate.mode == AgentMode.WAITING_FOR_INPUT
    store.submit(report('SubagentStop', **agent_fields))
    assert monitor.snapshot().aggregate.mode == AgentMode.COMPLETED
    assert store.pending_events('observer')[-1][1]['status']['state'] == 'done'


@pytest.mark.parametrize('initial,fields,expected', [
    ('SessionStart', {}, AgentMode.IDLE_READY),
    ('Interrupt', {}, AgentMode.IDLE_READY),
    ('UserPromptSubmit', {}, AgentMode.WAITING_FOR_INPUT),
    ('SessionStart', {'sidepulse_status': 'waiting_for_input'}, AgentMode.WAITING_FOR_INPUT),
    ('Stop', {}, AgentMode.COMPLETED),
    ('PermissionRequest', {}, AgentMode.WAITING_FOR_INPUT),
])
def test_idle_reminder_preserves_prior_state_without_hiding_explicit_wait(runtime, initial, fields, expected):
    store, monitor = runtime
    store.submit(report(initial))
    store.submit(report('Notification', notification_type='idle_prompt', message='Waiting for your input', **fields))
    assert monitor.snapshot().aggregate.mode == expected
    assert store.pending_events('observer')[-1][1]['status']['state'] == {
        AgentMode.IDLE_READY: 'idle', AgentMode.WAITING_FOR_INPUT: 'blocked', AgentMode.COMPLETED: 'done',
    }[expected]


def prepare_idle_metadata(runtime, tmp_path, monkeypatch, **overrides):
    store, _ = runtime
    now = datetime.now(timezone.utc)
    prompt_at, idle_at = now - timedelta(seconds=8), now - timedelta(seconds=1)
    with patch('sidepulse.program_status.datetime') as clock:
        clock.now.return_value = prompt_at
        store.submit(report('UserPromptSubmit'))
    directory = tmp_path / '.claude' / 'sessions'
    directory.mkdir(parents=True)
    value = {'sessionId': 'local-parent', 'pid': os.getpid(), 'procStart': 'valid-start',
             'kind': 'interactive', 'status': 'idle', 'statusUpdatedAt': int(idle_at.timestamp() * 1000)}
    value.update(overrides)
    path = directory / f'{os.getpid()}.json'
    path.write_text(json.dumps(value))
    monkeypatch.setattr('sidepulse.claude_session_state.session_directories', lambda: (directory,))
    monkeypatch.setattr('sidepulse.claude_session_state.process_matches', lambda pid, start: start == 'valid-start')
    return path, idle_at


def test_explicit_idle_metadata_reaches_daemon_snapshot_and_delivery(runtime, tmp_path, monkeypatch):
    store, monitor = runtime
    _, idle_at = prepare_idle_metadata(runtime, tmp_path, monkeypatch)
    before = store.statuses()[0].seq
    status, = monitor.snapshot().statuses
    assert (status.mode, status.event_name) == (AgentMode.IDLE_READY, 'SessionIdle')
    assert int(status.updated_at.timestamp() * 1000) == int(idle_at.timestamp() * 1000)
    assert status.seq == before + 1
    events = store.pending_events('observer')
    assert events[-1][1]['status']['state'] == 'idle'
    monitor.snapshot()
    assert store.pending_events('observer') == events


@pytest.mark.parametrize('overrides', [
    {'status': 'working'}, {'kind': 'background'}, {'procStart': 'reused-pid'},
    {'statusUpdatedAt': 0}, {'statusUpdatedAt': 9999999999999}, {'statusUpdatedAt': 'invalid'},
])
def test_invalid_idle_evidence_cannot_clear_daemon_work(runtime, tmp_path, monkeypatch, overrides):
    _, monitor = runtime
    prepare_idle_metadata(runtime, tmp_path, monkeypatch, **overrides)
    assert monitor.snapshot().aggregate.mode == AgentMode.WORKING


@pytest.mark.parametrize('fields', [{'agent_id': 'child'}, {'hook_event_name': 'PreToolUse'}, {'provider': 'codex'}])
def test_idle_reconciliation_only_targets_parent_claude_prompts(runtime, monkeypatch, fields):
    store, monitor = runtime
    payload = {'hook_event_name': 'UserPromptSubmit', 'session_id': 'local-parent', **fields}
    provider = payload.pop('provider', 'claude')
    with patch('sidepulse.program_status.datetime') as clock:
        clock.now.return_value = datetime.now(timezone.utc) - timedelta(seconds=8)
        store.submit(hook_report(provider, payload))
    with patch('sidepulse.claude_session_state.confirmed_idle_at') as read:
        monitor.snapshot()
    read.assert_not_called()


def test_remote_claude_status_never_reads_local_metadata(runtime):
    store, monitor = runtime
    source = ProgramState()
    try:
        with patch('sidepulse.program_status.datetime') as clock:
            clock.now.return_value = datetime.now(timezone.utc) - timedelta(seconds=8)
            source.submit(report('UserPromptSubmit'))
        for _, update in source.pending_events('receiver'):
            store.receive_update('remote', update)
        with patch('sidepulse.claude_session_state.confirmed_idle_at') as read:
            assert monitor.snapshot().aggregate.mode == AgentMode.WORKING
        read.assert_not_called()
    finally:
        source.close()


def test_new_hook_wins_race_with_idle_metadata_read(runtime, tmp_path, monkeypatch):
    store, monitor = runtime
    _, idle_at = prepare_idle_metadata(runtime, tmp_path, monkeypatch)
    def concurrent_tool(*args, **kwargs):
        store.submit(report('PreToolUse', tool_name='Bash'))
        return idle_at
    monkeypatch.setattr('sidepulse.claude_session_state.confirmed_idle_at', concurrent_tool)
    status, = monitor.snapshot().statuses
    assert status.event_name == 'PreToolUse'
    assert status.mode == AgentMode.TOOL_RUNNING


def test_local_socket_preserves_idle_reminder_and_clears_subagent_wait(runtime):
    from sidepulse.program_transport import ProgramServer, daemon_command, submit_report
    store, _ = runtime
    with tempfile.TemporaryDirectory(prefix='sp-local-', dir=None if os.name == 'nt' else '/tmp') as directory:
        path = Path(directory) / 'status.sock'
        server = ProgramServer(store, path)
        server.start()
        try:
            for event, fields in [('SessionStart', {}), ('Notification', {'notification_type': 'idle_prompt', 'message': 'Waiting for your input'})]:
                submit_report(report(event, **fields), path=path)
            snapshot = daemon_command({'type': 'get_state'}, path=path)['snapshot']
            assert snapshot['aggregate']['mode'] == 'idle_ready'
            submit_report(report('PermissionRequest', agent_id='child', tool_name='Bash'), path=path)
            snapshot = daemon_command({'type': 'get_state'}, path=path)['snapshot']
            assert snapshot['aggregate']['mode'] == 'waiting_for_input'
            submit_report(report('SubagentStop', agent_id='child'), path=path)
            snapshot = daemon_command({'type': 'get_state'}, path=path)['snapshot']
            assert snapshot['aggregate']['mode'] == 'completed'
        finally:
            server.stop()


@pytest.mark.parametrize('provider', ['claude', 'codex'])
def test_subagent_completion_preserves_parent_and_other_child_approvals(runtime, provider):
    store, monitor = runtime
    store.submit(report('PermissionRequest', provider, tool_name='Bash', tool_use_id='parent'))
    store.submit(report('PermissionRequest', provider, agent_id='first-child', tool_name='Bash', tool_use_id='first'))
    store.submit(report('PermissionRequest', provider, agent_id='second-child', tool_name='Bash', tool_use_id='second'))
    store.submit(report('SubagentStop', provider, agent_id='second-child'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WAITING_FOR_INPUT
    assert store.pending_events('observer')[-1][1]['context']['agent_id'] == 'first-child'
    store.submit(report('SubagentStop', provider, agent_id='first-child'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WAITING_FOR_INPUT
    assert 'agent_id' not in store.pending_events('observer')[-1][1]['context']
    store.submit(report('PostToolUse', provider, tool_name='Bash', tool_use_id='parent'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WORKING
    store.submit(report('SubagentStop', provider, agent_id='first-child'))
    assert monitor.snapshot().aggregate.mode == AgentMode.COMPLETED


def test_cleared_session_cannot_resurrect_old_parent_approval(runtime):
    store, monitor = runtime
    pending = report('PermissionRequest', tool_name='Bash')
    store.submit(pending)
    store.submit({'v': 1, 'type': 'program_status', 'scope': pending['scope'], 'status': {'state': 'clear'}})
    store.submit(report('SubagentStop', agent_id='child'))
    assert monitor.snapshot().aggregate.mode == AgentMode.COMPLETED


@pytest.mark.parametrize('provider', ['claude', 'codex'])
def test_subagent_stop_question_still_waits(runtime, provider):
    store, monitor = runtime
    store.submit(report('PermissionRequest', provider, agent_id='child', tool_name='Bash'))
    store.submit(report('SubagentStop', provider, agent_id='child', last_assistant_message='Which option should I use?'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WAITING_FOR_INPUT
    assert store.pending_events('observer')[-1][1]['status']['kind'] == 'question'


def test_session_end_releases_all_owned_approvals(runtime):
    store, monitor = runtime
    store.submit(report('PermissionRequest', agent_id='child', tool_name='Bash'))
    store.submit(report('SessionEnd'))
    assert monitor.snapshot().aggregate.mode == AgentMode.COMPLETED
    store.submit(report('UserPromptSubmit'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WORKING


def test_child_error_keeps_priority_over_other_owner_permission(runtime):
    store, monitor = runtime
    store.submit(report('PermissionRequest', tool_name='Bash'))
    store.submit(report('PostToolUseFailure', agent_id='child', message='Tool failed'))
    assert monitor.snapshot().aggregate.mode == AgentMode.BLOCKED_ERROR
    store.submit(report('SubagentStop', agent_id='child'))
    assert monitor.snapshot().aggregate.mode == AgentMode.WAITING_FOR_INPUT


@pytest.mark.parametrize('action', [[], {}, False, 'invalid'])
def test_invalid_permission_marker_does_not_mutate_state(runtime, action):
    store, _ = runtime
    store.submit(report('PermissionRequest', tool_name='Bash'))
    before = store.statuses(), store.pending_events('observer')
    with pytest.raises(StatusError, match='Unknown hook permission action'):
        store.submit({**report('SubagentStop'), '_hook_permission_action': action})
    assert (store.statuses(), store.pending_events('observer')) == before
