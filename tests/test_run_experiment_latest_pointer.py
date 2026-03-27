from pathlib import Path

import importlib.util


def _load_module():
    script = Path(__file__).parent.parent / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_experiment module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_update_latest_pointer(tmp_path: Path) -> None:
    mod = _load_module()
    reports = tmp_path / "reports"
    pipeline_dir = reports / "pipeline_20260101T000000Z"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "metadata.json").write_text('{"ok": true}', encoding="utf-8")
    mod._update_latest_pointer(reports, pipeline_dir)
    assert (reports / "latest" / "LATEST_PIPELINE.txt").exists()
    assert (reports / "latest" / "LATEST_METADATA.json").exists()
