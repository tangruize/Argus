"""Synthetic offline checks for integer Copilot budget observations."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.adapters.agent_cli_backend import _budget_monitor as budgets
from argus.core.cost_control import cost_control_snapshot, reserve_call_budget
from argus.provider_integrations.copilot_usage import CopilotCallUsage, CopilotModelUsage

# Generated amounts, not provider records: float addition overshoots by two ULPs.
NANOS = (11_000_000_000,) * 13
NOW = 1_800_000_000.0


def _rows(amounts: tuple[int, ...] = NANOS) -> tuple[CopilotModelUsage, ...]:
    return tuple(CopilotModelUsage(
        row_id=index, session_id="synthetic-session", turn_index=index,
        model="synthetic-a" if index % 2 else "synthetic-b",
        input_tokens=10, output_tokens=2, cache_read_tokens=8,
        cache_write_tokens=None, reasoning_tokens=1, total_nano_aiu=amount,
        request_multiplier=None, created_at=datetime.fromtimestamp(NOW, UTC).isoformat(),
    ) for index, amount in enumerate(amounts))


def _monitor(monkeypatch: pytest.MonkeyPatch, rows: tuple[CopilotModelUsage, ...]):
    monkeypatch.setattr(budgets.time, "time", lambda: NOW)
    reader = Mock(return_value=CopilotCallUsage(rows))
    monkeypatch.setattr(budgets, "read_copilot_usage_since", reader)
    reservation = SimpleNamespace(observe_cost=Mock(return_value=""))
    ctx = SimpleNamespace(
        resume_thread_id=None, provider_session_id="synthetic-session",
        backend=SimpleNamespace(_is_copilot=True), cost_reservation=reservation,
        copilot_usage_cursor=object(),
    )
    return budgets.LiveBudgetMonitor(ctx, interval_seconds=0), reservation, reader


@pytest.mark.parametrize("order", ["forward", "reverse", "interleaved"])
def test_copilot_observation_matches_canonical_integer_sum(monkeypatch, order):
    # Also vary the amounts while preserving the synthetic total.
    amounts = (10_000_000_000, 12_000_000_000) + NANOS[2:]
    rows = _rows(amounts)
    if order == "reverse":
        rows = rows[::-1]
    elif order == "interleaved":
        rows = rows[::2] + rows[1::2]
    monitor, reservation, reader = _monitor(monkeypatch, rows)
    # Growing query results are cumulative since the cursor, not increments.
    for end in (4, 9, 13, 13):
        reader.return_value = CopilotCallUsage(rows[:end])
        assert monitor.check() is None
        reservation.observe_cost.assert_called_with(
            reader.return_value.cost_usd, tokens=end * 13,
        )
    assert reader.call_args.kwargs == {"session_id": "synthetic-session", "timeout": 0}


def test_thirteen_synthetic_rows_do_not_inflate_observed_cost(monkeypatch):
    rows = _rows()
    sequential = 0.0
    for amount in NANOS:
        sequential += amount / 100_000_000_000
    assert sequential == 1.4300000000000004
    monitor, reservation, _reader = _monitor(monkeypatch, rows)
    assert monitor.check() is None
    assert CopilotCallUsage(rows).cost_usd == 1.43
    reservation.observe_cost.assert_called_once_with(1.43, tokens=169)


def test_optional_cost_and_date_filter_preserve_known_usage(monkeypatch):
    known = _rows()
    missing = replace(known[0], total_nano_aiu=None, input_tokens=None,
                      output_tokens=3, reasoning_tokens=None)
    old = replace(known[0], total_nano_aiu=100_000_000_000_000,
                  created_at=datetime.fromtimestamp(NOW - 172800, UTC).isoformat())
    undated = replace(old, created_at="")
    monitor, reservation, _reader = _monitor(monkeypatch, known + (missing, old, undated))
    assert monitor.check() is None
    # Missing prices remain missing; count known cost and raw tokens, not cache twice.
    assert CopilotCallUsage(known + (missing,)).cost_usd is None
    reservation.observe_cost.assert_called_once_with(1.43, tokens=172)


@pytest.mark.parametrize("limit", ["cost", "tokens"])
def test_lower_repeated_observations_cannot_erase_real_budget_obligations(
    tmp_path, monkeypatch, limit,
):
    monitor, _mock_reservation, reader = _monitor(monkeypatch, _rows())
    monkeypatch.setenv("ARGUS_SKILL_GLOBAL_DAILY_TOKEN_CAP", "169" if limit == "tokens" else "0")
    monkeypatch.setenv("ARGUS_SKILL_GLOBAL_DAILY_CAP_USD", "1.43" if limit == "cost" else "100")
    reservation, reason = reserve_call_budget(
        call_id="synthetic-call", project_root=None, mission_id=None,
        provider="copilot", model="synthetic-a", run_label="test",
        global_root=tmp_path, global_daily_cap_usd=1.43 if limit == "cost" else 100,
    )
    assert reservation is not None and not reason
    monitor.ctx.cost_reservation = reservation
    try:
        assert "budget exhausted" in (monitor.check() or "")
        # A subsequent genuinely underreported cost/token count cannot reopen admission.
        assert "budget exhausted" in reservation.observe_cost(1.42, tokens=168)
        snapshot = cost_control_snapshot(global_root=tmp_path)
        assert snapshot["in_flight_cost_usd"] == 1.43
        assert snapshot["daily_tokens"] == 169
        reader.return_value = CopilotCallUsage(())
        assert "budget exhausted" in (monitor.check() or "")
    finally:
        reservation.release(reason="synthetic test cleanup")
