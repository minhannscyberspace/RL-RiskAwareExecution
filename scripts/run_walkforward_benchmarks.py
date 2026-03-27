from __future__ import annotations

import numpy as np

from rl_riskaware.evaluation import make_walkforward_windows, run_window_benchmarks


def main() -> None:
    total_len = 240
    prices = np.linspace(100.0, 102.0, total_len, dtype=np.float64) + 0.2 * np.sin(
        np.linspace(0.0, 10.0 * np.pi, total_len)
    )
    volumes = (900_000.0 + 250_000.0 * (1.0 + np.sin(np.linspace(0.0, 8.0 * np.pi, total_len)))).astype(np.float64)

    windows = make_walkforward_windows(total_len=total_len, train_len=80, test_len=40, step=40)
    results = run_window_benchmarks(
        prices=prices,
        volumes=volumes,
        order_size=30_000.0,
        participation_cap=0.01,
        impact_eta=0.05,
        fixed_fee=0.0001,
        terminal_penalty=1.0,
        windows=windows,
    )

    print("window,policy,is,completion,avg_exec_price,slippage_bps")
    for r in results:
        print(
            f"{r.window_id},{r.policy},{r.is_value:.6f},{r.completion:.4f},{r.avg_exec_price:.6f},{r.slippage_bps_value:.3f}"
        )


if __name__ == "__main__":
    main()
