from rl_riskaware.evaluation import aggregate_rows_by_keys


def test_aggregate_rows_by_keys() -> None:
    rows = [
        {"algo": "ppo", "policy": "PPO", "completion": 1.0, "is": 1.0, "avg_exec_price": 100.0, "slippage_bps": 1.0},
        {"algo": "ppo", "policy": "PPO", "completion": 0.5, "is": 3.0, "avg_exec_price": 102.0, "slippage_bps": 3.0},
        {"algo": "sac", "policy": "SAC", "completion": 0.8, "is": 2.0, "avg_exec_price": 101.0, "slippage_bps": 2.0},
    ]
    out = aggregate_rows_by_keys(rows, group_keys=("algo", "policy"))
    keyed = {(r["algo"], r["policy"]): r for r in out}
    assert abs(float(keyed[("ppo", "PPO")]["completion_mean"]) - 0.75) < 1e-12
    assert keyed[("sac", "SAC")]["n_rows"] == 1


def test_aggregate_rows_by_keys_with_regime() -> None:
    rows = [
        {"regime": "calm", "policy": "PPO", "completion": 1.0, "is": 1.0, "avg_exec_price": 100.0, "slippage_bps": 1.0},
        {"regime": "calm", "policy": "PPO", "completion": 0.0, "is": 5.0, "avg_exec_price": 101.0, "slippage_bps": 2.0},
        {"regime": "volatile", "policy": "PPO", "completion": 0.5, "is": 3.0, "avg_exec_price": 102.0, "slippage_bps": 3.0},
    ]
    out = aggregate_rows_by_keys(rows, group_keys=("regime", "policy"))
    keyed = {(r["regime"], r["policy"]): r for r in out}
    assert keyed[("calm", "PPO")]["n_rows"] == 2
    assert abs(float(keyed[("calm", "PPO")]["completion_mean"]) - 0.5) < 1e-12
