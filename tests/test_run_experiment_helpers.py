import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).parent.parent / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_experiment module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_latest_run_dir(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "train_ppo_20200101T000000Z").mkdir()
    (tmp_path / "train_ppo_20220101T000000Z").mkdir()
    latest = mod._latest_run_dir(tmp_path, "train_ppo")
    assert latest.name == "train_ppo_20220101T000000Z"
