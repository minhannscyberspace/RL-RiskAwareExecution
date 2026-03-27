from rl_riskaware.evaluation import aggregate_policy_rows


def test_aggregate_policy_rows() -> None:
    rows = [
        {"window": 0, "policy": "PPO", "completion": 0.5, "is": 10.0, "avg_exec_price": 100.1, "slippage_bps": 1.0},
        {"window": 1, "policy": "PPO", "completion": 0.7, "is": 6.0, "avg_exec_price": 100.2, "slippage_bps": 2.0},
        {"window": 0, "policy": "TWAP", "completion": 1.0, "is": 8.0, "avg_exec_price": 100.3, "slippage_bps": 3.0},
    ]
    summary = aggregate_policy_rows(rows)
    by_policy = {r["policy"]: r for r in summary}
    assert by_policy["PPO"]["n_windows"] == 2
    assert abs(float(by_policy["PPO"]["completion_mean"]) - 0.6) < 1e-9
    assert by_policy["TWAP"]["n_windows"] == 1


def test_aggregate_policy_rows_with_sac_label() -> None:
    rows = [
        {"window": 0, "policy": "SAC", "completion": 1.0, "is": 5.0, "avg_exec_price": 100.0, "slippage_bps": 1.0},
        {"window": 1, "policy": "SAC", "completion": 0.0, "is": 9.0, "avg_exec_price": 101.0, "slippage_bps": 2.0},
    ]
    summary = aggregate_policy_rows(rows)
    assert len(summary) == 1
    assert summary[0]["policy"] == "SAC"
