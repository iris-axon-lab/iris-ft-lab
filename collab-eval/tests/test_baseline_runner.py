"""
Tests for the model baseline runner and train scaffold.

All tests run offline without GPU, model download, or API key.
Tests verify graceful degradation when model is unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))


class TestMLXRunnerGracefulDegradation:
    def test_check_mlx_available_returns_tuple(self):
        from collab_eval.inference.mlx_runner import check_mlx_available
        result = check_mlx_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        avail, reason = result
        assert isinstance(avail, bool)
        assert isinstance(reason, str)

    def test_check_mlx_available_reason_when_unavailable(self):
        """If mlx_lm is not installed, reason must be non-empty."""
        from collab_eval.inference.mlx_runner import check_mlx_available
        avail, reason = check_mlx_available()
        if not avail:
            assert reason, "Reason string must be non-empty when mlx_lm unavailable"

    def test_load_model_raises_runtime_error_when_unavailable(self):
        """load_model should raise RuntimeError (not ImportError) when mlx_lm absent."""
        from collab_eval.inference.mlx_runner import check_mlx_available, load_model
        avail, _ = check_mlx_available()
        if not avail:
            with pytest.raises(RuntimeError):
                load_model("mlx-community/Qwen2.5-3B-Instruct-4bit")
        else:
            pytest.skip("mlx_lm is installed; skipping unavailability test")

    def test_default_model_constant(self):
        from collab_eval.inference.mlx_runner import DEFAULT_MODEL
        assert DEFAULT_MODEL == "mlx-community/Qwen2.5-3B-Instruct-4bit"


class TestBaselineRunnerDryRun:
    def test_dry_run_exits_without_inference(self, tmp_path, capsys):
        from scripts.run_model_baseline import run
        output_path = str(tmp_path / "baseline_raw.jsonl")
        run(
            tasks_path=str(_ROOT / "data" / "generated" / "spreadsheet_heldout_v1.jsonl"),
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            limit=5,
            output_path=output_path,
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()
        # No raw output file written
        assert not Path(output_path).exists()

    def test_dry_run_prints_config(self, tmp_path, capsys):
        from scripts.run_model_baseline import run
        run(
            tasks_path="some/tasks.jsonl",
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            limit=10,
            output_path=str(tmp_path / "out.jsonl"),
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "mlx-community/Qwen2.5-3B-Instruct-4bit" in out
        assert "some/tasks.jsonl" in out


class TestBaselineRunnerModelUnavailable:
    def test_graceful_skip_when_mlx_unavailable(self, tmp_path, capsys, monkeypatch):
        """When mlx_lm is not available, runner prints message and exits 0."""
        import scripts.run_model_baseline as runner_mod
        monkeypatch.setattr(
            runner_mod,
            "check_mlx_available",
            lambda: (False, "mlx_lm not installed (test stub)"),
        )
        # Redirect _BASELINE_MD to tmp_path so the test doesn't write to the real file
        monkeypatch.setattr(runner_mod, "_BASELINE_MD", tmp_path / "model_baseline_v0.md")
        output_path = str(tmp_path / "baseline_raw.jsonl")
        runner_mod.run(
            tasks_path="data/generated/spreadsheet_heldout_v1.jsonl",
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            limit=5,
            output_path=output_path,
            dry_run=False,
        )
        out = capsys.readouterr().out
        assert "Model unavailable" in out
        assert "Skipping inference" in out
        # No raw JSONL written
        assert not Path(output_path).exists()

    def test_graceful_skip_writes_unavailable_md(self, tmp_path, capsys, monkeypatch):
        """Runner writes model_baseline_v0.md with unavailable message."""
        import scripts.run_model_baseline as runner_mod
        monkeypatch.setattr(
            runner_mod,
            "check_mlx_available",
            lambda: (False, "mlx_lm not installed (test stub)"),
        )
        # Redirect _BASELINE_MD to tmp_path
        orig_md = runner_mod._BASELINE_MD
        runner_mod._BASELINE_MD = tmp_path / "model_baseline_v0.md"
        try:
            runner_mod.run(
                tasks_path="data/generated/spreadsheet_heldout_v1.jsonl",
                model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
                limit=5,
                output_path=str(tmp_path / "raw.jsonl"),
                dry_run=False,
            )
        finally:
            runner_mod._BASELINE_MD = orig_md
        md_content = (tmp_path / "model_baseline_v0.md").read_text()
        assert "Baseline not yet run" in md_content or "Model unavailable" in md_content


class TestGradeCaseFunction:
    def test_grade_case_perfect_score_on_gold(self):
        """Gold output from generator should score highly on grade_case."""
        from collab_eval.generation.spreadsheet_generator import generate_cases
        from scripts.run_model_baseline import grade_case
        import dataclasses
        cases = generate_cases(n=5, seed=42)
        for case in cases:
            case_dict = dataclasses.asdict(case)
            gold = case.gold_or_reference_output
            scores, flags, composite = grade_case(case_dict, gold)
            assert composite >= 0.9, (
                f"{case.case_id}: gold output scored {composite:.3f} < 0.9"
            )
            assert "csv_not_parseable" not in flags

    def test_grade_case_empty_output_fails(self):
        from collab_eval.generation.spreadsheet_generator import generate_cases
        from scripts.run_model_baseline import grade_case
        import dataclasses
        cases = generate_cases(n=2, seed=42)
        for case in cases:
            case_dict = dataclasses.asdict(case)
            scores, flags, composite = grade_case(case_dict, "")
            assert "csv_not_parseable" in flags
            assert composite <= 0.3

    def test_grade_case_uses_case_metadata(self):
        """grade_case must use expected_row_count from case metadata, not a hardcoded value."""
        from collab_eval.generation.spreadsheet_generator import generate_cases
        from scripts.run_model_baseline import grade_case
        import dataclasses
        # Generate a case with a specific row count and verify grading respects it
        cases = generate_cases(n=10, seed=42)
        for case in cases:
            case_dict = dataclasses.asdict(case)
            expected_rows = case_dict["expected_metadata"]["expected_row_count"]
            gold = case.gold_or_reference_output
            scores, flags, composite = grade_case(case_dict, gold)
            # Gold output should always match its own metadata
            assert scores["data_preservation"] == 1.0, (
                f"{case.case_id}: expected data_preservation=1.0 on gold, "
                f"got {scores['data_preservation']} (expected_rows={expected_rows})"
            )


class TestTrainScaffoldGracefulBehavior:
    def test_check_only_exits_gracefully_without_mlx(self, capsys, monkeypatch):
        from scripts import train_collab_sft as train_mod
        monkeypatch.setattr(
            train_mod,
            "_check_dependencies",
            lambda: (False, ["mlx_lm not installed (test stub)"]),
        )
        train_mod.run_check_only("configs/sft_collab_eval_qwen25_3b.yaml")
        out = capsys.readouterr().out
        assert "NOT AVAILABLE" in out or "not available" in out.lower()

    def test_dry_run_exits_gracefully_without_mlx(self, capsys, monkeypatch):
        from scripts import train_collab_sft as train_mod
        monkeypatch.setattr(
            train_mod,
            "_check_dependencies",
            lambda: (False, ["mlx_lm not installed (test stub)"]),
        )
        train_mod.run_dry_run("configs/sft_collab_eval_qwen25_3b.yaml")
        out = capsys.readouterr().out
        assert "NOT AVAILABLE" in out or "not available" in out.lower()

    def test_dry_run_with_mlx_prints_command(self, capsys, monkeypatch):
        from scripts import train_collab_sft as train_mod
        monkeypatch.setattr(
            train_mod,
            "_check_dependencies",
            lambda: (True, []),
        )
        train_mod.run_dry_run(
            str(_ROOT / "configs" / "sft_collab_eval_qwen25_3b.yaml")
        )
        out = capsys.readouterr().out
        assert "mlx_lm.lora" in out
        assert "--config" in out
        assert "sft_collab_eval_qwen25_3b.yaml" in out
        assert "not executed" in out.lower() or "dry-run" in out.lower()

    def test_no_action_prints_help(self, capsys, monkeypatch):
        from scripts import train_collab_sft as train_mod
        monkeypatch.setattr(train_mod, "_check_dependencies", lambda: (True, []))
        # Test the run function with no flags set
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train_mod.run_check_only(
                str(_ROOT / "configs" / "sft_collab_eval_qwen25_3b.yaml")
            )
        output = buf.getvalue()
        # check-only should not raise even if mlx_lm is not installed
        assert len(output) > 0


class TestNoTrainInTests:
    def test_baseline_runner_does_not_train(self):
        """Smoke test: importing run_model_baseline doesn't trigger inference."""
        import scripts.run_model_baseline  # noqa: F401
        # No exception = pass

    def test_train_scaffold_does_not_train(self):
        """Smoke test: importing train_collab_sft doesn't trigger training."""
        import scripts.train_collab_sft  # noqa: F401
        # No exception = pass

    def test_eval_scaffold_does_not_train(self):
        """Smoke test: importing run_collab_model_eval doesn't trigger inference."""
        import eval.run_collab_model_eval  # noqa: F401
        # No exception = pass
