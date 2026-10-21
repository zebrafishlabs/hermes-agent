"""An opt-in required pre-run script must fail before prompt/agent construction."""
from unittest.mock import Mock
import json
import pytest
from cron import scheduler as S


@pytest.fixture
def gates(monkeypatch):
    import hermes_cli.config as config
    monkeypatch.setattr(config, "require_parseable_user_config", lambda: None)
    monkeypatch.setattr(S, "_apply_monitor_gate", lambda *a: (None, None, None))
    monkeypatch.setattr(S, "_resolve_job_workdir", lambda *a: None)
    build = Mock(return_value="SYNTHETIC prompt")
    monkeypatch.setattr(S, "_build_job_prompt", build)
    return build


def prepare(gates, monkeypatch, required, success, output="SYNTHETIC script error", script="gate.py"):
    run = Mock(return_value=(success, output))
    monkeypatch.setattr(S, "_run_job_script_with_claim_heartbeat", run)
    job = {"id": "synthetic", "name": "synthetic", "prompt": "Inspect only", "script": script}
    if required is not None: job["require_script_success"] = required
    return S._prepare_job_prompt(job, job["id"], job["name"], None, None), run


def test_required_failure_stops_before_prompt(gates, monkeypatch):
    (early, prompt), _ = prepare(gates, monkeypatch, True, False)
    assert early is not None and early[0] is False and prompt is None
    assert "SYNTHETIC script error" in early[3]
    gates.assert_not_called()


@pytest.mark.parametrize("required", [False, None])
def test_advisory_behavior_unchanged(gates, monkeypatch, required):
    (early, prompt), _ = prepare(gates, monkeypatch, required, False)
    assert early is None and prompt == "SYNTHETIC prompt"
    gates.assert_called_once()


def test_required_success_continues(gates, monkeypatch):
    (early, prompt), _ = prepare(gates, monkeypatch, True, True, "verified")
    assert early is None and prompt == "SYNTHETIC prompt"


def test_required_missing_script_fails(gates, monkeypatch):
    (early, prompt), run = prepare(gates, monkeypatch, True, False, script=None)
    assert early is not None and early[0] is False and prompt is None
    run.assert_not_called(); gates.assert_not_called()


@pytest.mark.parametrize("required", ["false", 1, []])
def test_malformed_required_setting_fails_closed(gates, monkeypatch, required):
    (early, prompt), run = prepare(gates, monkeypatch, required, True)
    assert early is not None and early[0] is False and prompt is None
    run.assert_not_called(); gates.assert_not_called()


def test_cli_preserves_required_script_choice():
    import argparse
    from hermes_cli.subcommands.cron import build_cron_parser
    from hermes_cli.cron import _job_api_kwargs
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda args: None)
    for flag, expected in [("--require-script-success", True), ("--no-require-script-success", False)]:
        args = parser.parse_args(["cron", "edit", "synthetic", flag])
        assert _job_api_kwargs(args)["require_script_success"] is expected


def test_storage_and_tool_update_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from cron import jobs
    from tools.cronjob_tools import cronjob, _cronjob_handler
    job = jobs.create_job("Synthetic", "every 1h", script="gate.py", paused=True, require_script_success=True)
    assert jobs.get_job(job["id"])["require_script_success"] is True
    result = json.loads(cronjob("update", job_id=job["id"], require_script_success=False))
    assert result["success"]
    assert jobs.get_job(job["id"])["require_script_success"] is False
    with pytest.raises(ValueError): jobs.update_job(job["id"], {"require_script_success": "false"})
    with pytest.raises(ValueError): jobs.update_job(job["id"], {"require_script_success": None})
    result = json.loads(_cronjob_handler({"action": "update", "job_id": job["id"], "require_script_success": True}))
    assert result["success"] and result["job"]["require_script_success"] is True
    with pytest.raises(ValueError): jobs.update_job(job["id"], {"script": None})
