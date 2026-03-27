from pathlib import Path

from rl_riskaware.config import load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("a: 1\nb: test\n", encoding="utf-8")
    cfg = load_yaml_config(str(cfg_file))
    assert cfg["a"] == 1
    assert cfg["b"] == "test"
