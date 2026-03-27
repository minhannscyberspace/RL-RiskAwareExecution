from pathlib import Path

from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_metadata_json, write_results_csv


def test_artifact_writers_create_files(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, "unit")
    cfg_path = write_config_snapshot(run_dir, {"a": 1, "b": "x"})
    meta_path = write_metadata_json(run_dir, {"ok": True})
    csv_path = write_results_csv(run_dir, "results.csv", [{"k": "v", "n": 1}])

    assert cfg_path.exists()
    assert meta_path.exists()
    assert csv_path.exists()
