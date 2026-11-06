"""Offline tests for cloud/lambda_client.py and cloud/ledger.py (no network)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
import lambda_client  # noqa: E402
from ledger import Ledger, cost_so_far  # noqa: E402

TOKEN = "sekrit-token-123"


def test_request_headers():
    req = lambda_client.build_request("GET", "/instances", TOKEN)
    assert req.get_header("Authorization") == f"Bearer {TOKEN}"
    assert "Mozilla" in req.get_header("User-agent")
    assert req.full_url.startswith("https://cloud.lambda.ai/api/v1/")


def test_post_body_is_json():
    req = lambda_client.build_request("POST", "/x", TOKEN, {"a": 1})
    assert req.data == b'{"a": 1}'
    assert req.get_header("Content-type") == "application/json"


def test_cost_arithmetic():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hours, dollars = cost_so_far(1.99, t0.isoformat(), now=t0 + timedelta(hours=1.5))
    assert hours == pytest.approx(1.5)
    assert dollars == pytest.approx(2.985)


def test_ledger_roundtrip(tmp_path):
    led = Ledger(tmp_path)
    led.record_launch("i-1", "gpu_1x_a100_sxm4", "us-east-1", 1.99)
    led.record_status("i-1", "active", "1.2.3.4")
    rec = led.record_terminate("i-1")
    assert rec["ip"] == "1.2.3.4" and rec["final_cost"] is not None
    assert (tmp_path / "mnist-ddpm-rf.md").exists()
    assert len(led.read()["instances"]["i-1"]["events"]) == 3


def test_terminate_requires_yes(monkeypatch, capsys):
    monkeypatch.setenv("LAMBDA_API_KEY", TOKEN)
    with mock.patch.object(lambda_client.urllib.request, "urlopen") as uo:
        rc = lambda_client.main(["terminate", "--id", "i-1"])
    assert rc == 1
    uo.assert_not_called()
    assert "--yes" in capsys.readouterr().err


def test_token_never_printed(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("LAMBDA_API_KEY", TOKEN)
    body = b'{"data": [{"id": "i-1", "status": "active", "ip": "1.1.1.1"}]}'
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = body
    with mock.patch.object(lambda_client.urllib.request, "urlopen", return_value=resp):
        assert lambda_client.main(["instances"]) == 0
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err


def test_missing_token_errors(monkeypatch, capsys):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    assert lambda_client.main(["instances"]) == 1
