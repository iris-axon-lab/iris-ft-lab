"""
Smoke tests for YAML config files.

Tests:
  - All config files are valid YAML
  - SFT config has required keys with sane values
  - DPO config has required keys
  - Model registry lists expected models with required fields
"""

import os
import sys

import pytest
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def load_yaml(filename):
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── SFT config ────────────────────────────────────────────────────────────────

class TestSFTConfig:
    def setup_method(self):
        self.config = load_yaml("sft_trace_qwen25_3b.yaml")

    def test_loads_without_error(self):
        assert self.config is not None

    def test_has_model_path(self):
        assert "model" in self.config
        assert "path" in self.config["model"]
        assert isinstance(self.config["model"]["path"], str)
        assert len(self.config["model"]["path"]) > 0

    def test_has_lora_section(self):
        assert "lora" in self.config
        lora = self.config["lora"]
        assert "rank" in lora
        assert "alpha" in lora
        assert "target_modules" in lora

    def test_lora_rank_is_positive_int(self):
        assert isinstance(self.config["lora"]["rank"], int)
        assert self.config["lora"]["rank"] > 0

    def test_lora_alpha_is_positive(self):
        alpha = self.config["lora"]["alpha"]
        assert isinstance(alpha, (int, float))
        assert alpha > 0

    def test_lora_target_modules_is_list(self):
        modules = self.config["lora"]["target_modules"]
        assert isinstance(modules, list)
        assert len(modules) >= 1

    def test_training_section_present(self):
        assert "training" in self.config
        training = self.config["training"]
        assert "epochs" in training
        assert "batch_size" in training
        assert "learning_rate" in training

    def test_epochs_is_positive(self):
        assert self.config["training"]["epochs"] >= 1

    def test_batch_size_is_positive(self):
        assert self.config["training"]["batch_size"] >= 1

    def test_learning_rate_in_reasonable_range(self):
        lr = self.config["training"]["learning_rate"]
        assert 1e-6 <= lr <= 1e-2, f"Learning rate {lr} looks suspicious"

    def test_data_section_present(self):
        assert "data" in self.config
        assert "train" in self.config["data"]
        assert "valid" in self.config["data"]

    def test_output_section_present(self):
        assert "output" in self.config
        assert "adapter_path" in self.config["output"]

    def test_model_path_not_hardcoded_local(self):
        # Model paths should be HF repo IDs or config-level references,
        # not absolute system paths like /Users/...
        path = self.config["model"]["path"]
        assert not path.startswith("/Users/"), (
            "Model path looks like a hardcoded personal path. "
            "Use an HF repo ID or a path variable instead."
        )


# ── DPO config ────────────────────────────────────────────────────────────────

class TestDPOConfig:
    def setup_method(self):
        self.config = load_yaml("dpo_trace_qwen25_3b.yaml")

    def test_loads_without_error(self):
        assert self.config is not None

    def test_has_model_path(self):
        assert "model" in self.config
        assert "path" in self.config["model"]

    def test_has_dpo_section(self):
        assert "dpo" in self.config
        assert "beta" in self.config["dpo"]

    def test_dpo_beta_in_valid_range(self):
        beta = self.config["dpo"]["beta"]
        assert 0 < beta <= 1.0, f"DPO beta={beta} is outside expected range (0, 1]"

    def test_has_data_section(self):
        assert "data" in self.config
        assert "train" in self.config["data"]

    def test_has_output_section(self):
        assert "output" in self.config
        assert "adapter_path" in self.config["output"]


# ── Model registry ────────────────────────────────────────────────────────────

class TestModelRegistry:
    def setup_method(self):
        self.registry = load_yaml("model_registry.yaml")

    def test_loads_without_error(self):
        assert self.registry is not None

    def test_has_models_list(self):
        assert "models" in self.registry
        assert isinstance(self.registry["models"], list)
        assert len(self.registry["models"]) >= 1

    def test_each_model_has_required_fields(self):
        required = {"alias", "path", "status", "description"}
        for model in self.registry["models"]:
            missing = required - model.keys()
            assert not missing, f"Model entry missing fields: {missing}"

    def test_expected_aliases_present(self):
        aliases = {m["alias"] for m in self.registry["models"]}
        expected = {"qwen25-3b", "qwen3-1.7b", "qwen25-7b", "qwen3-8b"}
        for alias in expected:
            assert alias in aliases, f"Expected alias '{alias}' not found in model registry"

    def test_primary_model_exists(self):
        primary = [m for m in self.registry["models"] if m.get("status") == "primary"]
        assert len(primary) >= 1, "Model registry should have at least one 'primary' model"

    def test_no_hardcoded_local_paths(self):
        for model in self.registry["models"]:
            path = model["path"]
            assert not path.startswith("/Users/"), (
                f"Model '{model['alias']}' has a hardcoded local path: {path}. "
                "Use an HF repo ID."
            )
