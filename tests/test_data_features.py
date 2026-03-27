from pathlib import Path

from rl_riskaware.data import load_market_csv
from rl_riskaware.features import build_lag_safe_features, prices_volumes_from_features


def test_load_market_csv_and_build_features() -> None:
    csv_path = Path(__file__).parent / "fixtures" / "market_small.csv"
    market = load_market_csv(str(csv_path))
    feats = build_lag_safe_features(market.df)
    prices, volumes = prices_volumes_from_features(feats.df)
    assert len(prices) == len(volumes)
    assert len(prices) > 0

