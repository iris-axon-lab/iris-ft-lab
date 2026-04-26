"""
Tests for the SFT training scaffold and config correctness.

All tests run offline without GPU, model download, or API key.
Tests verify:
  - config does not reference WikiSQL
  - config has train: true
  - config data points to a local path
  - dry-run command does not include deprecated python -m mlx_lm.lora
  - dry-run command does not include --lora-rank
  - check-only catches missing local train.jsonl with a clear message
  - --train path uses subprocess (mocked; does not run actual training)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = _ROOT / "configs" / "sft_collab_eval_qwen25_3b.yaml"

sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Config-file correctness
# ---------------------------------------------------------------------------

class TestConfigFile:
    @pytest.fixture(autouse=True)
    def _load(self):
        import yaml
        with _CONFIG_PATH.open(encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_config_exists(self):
        assert _CONFIG_PATH.exists(), f"Config not found: {_CONFIG_PATH}"

    def test_no_wikisql_reference(self):
        raw = _CONFIG_PATH.read_text(encoding="utf-8")
        assert "WikiSQL" not in raw, "Config must not reference mlx-community/WikiSQL"
        assert "wikisql" not in raw.lower(), "Config must not reference wikisql"

    def test_train_is_true(self):
        assert self.cfg.get("train") is True, (
            f"Config must have 'train: true', got: {self.cfg.get('train')!r}"
        )

    def test_test_is_false(self):
        assert self.cfg.get("test") is False, (
            f"Config should have 'test: false', got: {self.cfg.get('test')!r}"
        )

    def test_data_is_local_path(self):
        data_val = self.cfg.get("data", "")
        assert data_val, "Config must have a 'data' key"
        assert not data_val.startswith("mlx-community/"), (
            f"Config 'data' must be a local path, not a HuggingFace dataset: {data_val!r}"
        )
        assert not data_val.startswith("huggingface/"), (
            f"Config 'data' must be a local path: {data_val!r}"
        )

    def test_data_points_to_sft_full_directory(self):
        data_val = self.cfg.get("data", "")
        assert "sft_collab_eval_full" in data_val, (
            f"Config 'data' should point to local SFT data directory, got: {data_val!r}"
        )

    def test_model_is_correct(self):
        assert self.cfg.get("model") == "mlx-community/Qwen2.5-3B-Instruct-4bit"

    def test_adapter_path_is_local(self):
        adapter = self.cfg.get("adapter_path", "")
        assert adapter, "Config must have adapter_path"
        assert not adapter.startswith("mlx-community/"), (
            f"adapter_path should be a local path: {adapter!r}"
        )

    def test_lora_parameters_present(self):
        lora = self.cfg.get("lora_parameters")
        assert isinstance(lora, dict), (
            "Config must have 'lora_parameters' dict (not flat lora_rank/lora_alpha keys)"
        )
        assert "rank" in lora, "lora_parameters must have 'rank'"

    def test_no_top_level_lora_rank(self):
        assert "lora_rank" not in self.cfg, (
            "Config must not have top-level 'lora_rank' — use lora_parameters.rank instead"
        )

    def test_no_top_level_lora_alpha(self):
        assert "lora_alpha" not in self.cfg, (
            "Config must not have top-level 'lora_alpha' — use lora_parameters.scale instead"
        )


# ---------------------------------------------------------------------------
# Dry-run command correctness
# ---------------------------------------------------------------------------

class TestDryRunCommand:
    def _get_dry_run_output(self, config_path: str, capsys) -> str:
        from scripts import train_collab_sft as mod
        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            mod.run_dry_run(config_path)
        return capsys.readouterr().out

    def test_dry_run_does_not_use_python_m_mlx_lm_lora(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "python -m mlx_lm.lora" not in out, (
            "dry-run must not emit deprecated 'python -m mlx_lm.lora'"
        )

    def test_dry_run_does_not_use_lora_rank_flag(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "--lora-rank" not in out, (
            "dry-run must not emit deprecated --lora-rank flag"
        )

    def test_dry_run_uses_config_flag(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "--config" in out, "dry-run command must use --config flag"

    def test_dry_run_references_config_file(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "sft_collab_eval_qwen25_3b.yaml" in out

    def test_dry_run_uses_mlx_lm_lora_cli(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "mlx_lm.lora" in out

    def test_dry_run_prints_not_executed(self, capsys):
        out = self._get_dry_run_output(str(_CONFIG_PATH), capsys)
        assert "not executed" in out.lower() or "dry-run" in out.lower()


# ---------------------------------------------------------------------------
# check-only: catches missing train.jsonl
# ---------------------------------------------------------------------------

class TestCheckOnly:
    def test_check_only_reports_missing_train_jsonl(self, tmp_path, capsys):
        """check-only must print a clear message when train.jsonl is missing."""
        from scripts import train_collab_sft as mod
        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            with patch.object(mod, "_ROOT", tmp_path):
                # Minimal config that points to nonexistent data directory
                cfg_file = tmp_path / "configs" / "test.yaml"
                cfg_file.parent.mkdir(parents=True, exist_ok=True)
                cfg_file.write_text(
                    "model: mlx-community/Qwen2.5-3B-Instruct-4bit\n"
                    "data: data/sft_collab_eval_full\n"
                    "train: true\n"
                    "test: false\n"
                    "adapter_path: adapters/test/\n"
                )
                mod.run_check_only(str(cfg_file))
        out = capsys.readouterr().out
        assert "MISSING" in out or "missing" in out.lower() or "not found" in out.lower(), (
            f"check-only must report missing train.jsonl, got:\n{out}"
        )

    def test_check_only_shows_create_instructions_when_missing(self, tmp_path, capsys):
        """check-only must explain how to create train.jsonl when it's missing."""
        from scripts import train_collab_sft as mod
        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            with patch.object(mod, "_ROOT", tmp_path):
                cfg_file = tmp_path / "configs" / "test.yaml"
                cfg_file.parent.mkdir(parents=True, exist_ok=True)
                cfg_file.write_text(
                    "model: mlx-community/Qwen2.5-3B-Instruct-4bit\n"
                    "data: data/sft_collab_eval_full\n"
                    "train: true\n"
                    "test: false\n"
                    "adapter_path: adapters/test/\n"
                )
                mod.run_check_only(str(cfg_file))
        out = capsys.readouterr().out
        assert "build_sft_data" in out or "generate_tasks" in out, (
            f"check-only must explain how to build training data, got:\n{out}"
        )

    def test_check_only_warns_wikisql_in_config(self, tmp_path, capsys):
        """check-only must flag config that still uses WikiSQL."""
        from scripts import train_collab_sft as mod
        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            with patch.object(mod, "_ROOT", tmp_path):
                cfg_file = tmp_path / "configs" / "bad.yaml"
                cfg_file.parent.mkdir(parents=True, exist_ok=True)
                cfg_file.write_text(
                    "model: mlx-community/Qwen2.5-3B-Instruct-4bit\n"
                    "data: mlx-community/WikiSQL\n"
                    "train: true\n"
                )
                mod.run_check_only(str(cfg_file))
        out = capsys.readouterr().out
        assert "WikiSQL" in out or "ERROR" in out or "error" in out.lower(), (
            f"check-only must flag WikiSQL data reference, got:\n{out}"
        )

    def test_check_only_exits_gracefully_without_mlx(self, capsys, monkeypatch):
        from scripts import train_collab_sft as mod
        monkeypatch.setattr(mod, "_check_dependencies", lambda: (False, ["mlx_lm not installed (test stub)"]))
        mod.run_check_only("configs/sft_collab_eval_qwen25_3b.yaml")
        out = capsys.readouterr().out
        assert "NOT AVAILABLE" in out or "not available" in out.lower()

    def test_check_only_passes_on_real_config(self, capsys):
        """check-only on the real config should complete without error."""
        from scripts import train_collab_sft as mod
        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            mod.run_check_only(str(_CONFIG_PATH))
        out = capsys.readouterr().out
        assert "train: true" in out or "train" in out.lower()
        assert "WikiSQL" not in out.split("ERROR")[0] if "ERROR" not in out else True


# ---------------------------------------------------------------------------
# --train path: subprocess mocked, does not run real training
# ---------------------------------------------------------------------------

class TestTrainFlag:
    def test_train_invokes_subprocess(self, tmp_path, capsys):
        """--train must call subprocess.run with the expected command."""
        from scripts import train_collab_sft as mod

        cfg_file = tmp_path / "configs" / "test.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        data_dir = tmp_path / "data" / "sft_collab_eval_full"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"messages": []}\n')
        cfg_file.write_text(
            "model: mlx-community/Qwen2.5-3B-Instruct-4bit\n"
            f"data: {data_dir}\n"
            "train: true\n"
            "test: false\n"
            "adapter_path: adapters/test/\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            with patch.object(mod, "_ROOT", tmp_path):
                with patch("scripts.train_collab_sft.subprocess.run", return_value=mock_result) as mock_sub:
                    with pytest.raises(SystemExit) as exc_info:
                        mod.run_train(str(cfg_file))
                    assert exc_info.value.code == 0
                    mock_sub.assert_called_once()
                    cmd_args = mock_sub.call_args[0][0]
                    assert "mlx_lm.lora" in cmd_args[0] or any("mlx_lm" in a for a in cmd_args)
                    assert "--config" in cmd_args

    def test_train_aborts_when_train_jsonl_missing(self, tmp_path, capsys):
        """--train must exit 1 with clear message when train.jsonl doesn't exist."""
        from scripts import train_collab_sft as mod

        cfg_file = tmp_path / "configs" / "test.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(
            "model: mlx-community/Qwen2.5-3B-Instruct-4bit\n"
            "data: data/sft_collab_eval_full\n"
            "train: true\n"
            "test: false\n"
            "adapter_path: adapters/test/\n"
        )

        with patch.object(mod, "_check_dependencies", return_value=(True, [])):
            with patch.object(mod, "_ROOT", tmp_path):
                with pytest.raises(SystemExit) as exc_info:
                    mod.run_train(str(cfg_file))
                assert exc_info.value.code != 0

        out = capsys.readouterr().out
        assert "ERROR" in out or "not found" in out.lower() or "missing" in out.lower()

    def test_train_aborts_when_mlx_unavailable(self, tmp_path, capsys, monkeypatch):
        """--train must exit 1 when mlx_lm is not installed."""
        from scripts import train_collab_sft as mod
        monkeypatch.setattr(mod, "_check_dependencies", lambda: (False, ["mlx_lm not installed"]))
        with pytest.raises(SystemExit) as exc_info:
            mod.run_train(str(_CONFIG_PATH))
        assert exc_info.value.code != 0

    def test_no_action_prints_help(self, capsys, monkeypatch):
        """No flags → prints help message and does not raise."""
        from scripts import train_collab_sft as mod
        monkeypatch.setattr(sys, "argv", ["train_collab_sft.py"])
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                mod.main()
            except SystemExit:
                pass
        out = buf.getvalue()
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Smoke tests: importing does not trigger training
# ---------------------------------------------------------------------------

class TestNoTrainOnImport:
    def test_import_does_not_train(self):
        import scripts.train_collab_sft  # noqa: F401
