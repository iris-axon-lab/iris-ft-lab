"""
Smoke tests for scripts/prepare_data.py

Tests:
  - Schema validation accepts valid records
  - Schema validation rejects invalid records with useful messages
  - train/val split is deterministic and correct ratio
  - JSONL loading handles empty lines gracefully
"""

import json
import os
import sys
import tempfile

import pytest

# Make scripts/ importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prepare_data import (
    load_jsonl,
    validate_sft_record,
    validate_dpo_record,
    validate_eval_record,
    split_records,
    write_jsonl,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SFT = {
    "messages": [
        {"role": "system", "content": "You are a Trace extraction engine."},
        {"role": "user", "content": "Finished debugging the race condition."},
        {"role": "assistant", "content": '{"memory_tier": "episodic"}'},
    ]
}

VALID_DPO = {
    "prompt": "System prompt\n\nInput: Told Sarah I'd review her PR.",
    "chosen": '{"memory_tier": "prospective", "stated_intent": "Review PR"}',
    "rejected": '{"memory_tier": "episodic", "stated_intent": null}',
}

VALID_EVAL = {
    "id": "eval_001",
    "input": "Paired with Alex on the auth refactor.",
    "gold": {
        "memory_tier": "episodic",
        "emotional_valence": "positive",
        "stated_intent": None,
        "topic_cluster": "engineering",
    },
    "eval_tags": ["tier_classification"],
}


# ── SFT validation ────────────────────────────────────────────────────────────

def test_valid_sft_record_passes():
    validate_sft_record(VALID_SFT, 0, "test.jsonl")  # should not raise


def test_sft_missing_messages_key_raises():
    with pytest.raises(ValueError, match="missing keys"):
        validate_sft_record({"text": "hello"}, 0, "test.jsonl")


def test_sft_empty_messages_raises():
    with pytest.raises(ValueError, match="at least 2 items"):
        validate_sft_record({"messages": [{"role": "user", "content": "hi"}]}, 0, "test.jsonl")


def test_sft_message_missing_role_raises():
    bad_record = {"messages": [{"content": "hi"}, {"role": "assistant", "content": "ok"}]}
    with pytest.raises(ValueError, match="role"):
        validate_sft_record(bad_record, 0, "test.jsonl")


# ── DPO validation ────────────────────────────────────────────────────────────

def test_valid_dpo_record_passes():
    validate_dpo_record(VALID_DPO, 0, "test.jsonl")


def test_dpo_missing_prompt_raises():
    bad = {"chosen": "...", "rejected": "..."}
    with pytest.raises(ValueError, match="missing keys"):
        validate_dpo_record(bad, 0, "test.jsonl")


def test_dpo_empty_chosen_raises():
    bad = {**VALID_DPO, "chosen": ""}
    with pytest.raises(ValueError, match="non-empty string"):
        validate_dpo_record(bad, 0, "test.jsonl")


# ── Eval validation ───────────────────────────────────────────────────────────

def test_valid_eval_record_passes():
    validate_eval_record(VALID_EVAL, 0, "test.jsonl")


def test_eval_invalid_tier_raises():
    bad = {
        **VALID_EVAL,
        "gold": {**VALID_EVAL["gold"], "memory_tier": "mythological"},
    }
    with pytest.raises(ValueError, match="invalid memory_tier"):
        validate_eval_record(bad, 0, "test.jsonl")


def test_eval_invalid_valence_raises():
    bad = {
        **VALID_EVAL,
        "gold": {**VALID_EVAL["gold"], "emotional_valence": "ecstatic"},
    }
    with pytest.raises(ValueError, match="invalid emotional_valence"):
        validate_eval_record(bad, 0, "test.jsonl")


def test_eval_missing_gold_key_raises():
    bad = {**VALID_EVAL, "gold": {"memory_tier": "episodic"}}
    with pytest.raises(ValueError, match="gold missing keys"):
        validate_eval_record(bad, 0, "test.jsonl")


# ── Split ─────────────────────────────────────────────────────────────────────

def test_split_is_deterministic():
    records = [{"id": i} for i in range(20)]
    train1, val1 = split_records(records, 0.10, seed=42)
    train2, val2 = split_records(records, 0.10, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_split_ratio_approx_correct():
    records = [{"id": i} for i in range(100)]
    train, val = split_records(records, 0.10, seed=42)
    assert len(val) == 10
    assert len(train) == 90
    assert len(train) + len(val) == 100


def test_split_different_seeds_produce_different_results():
    records = [{"id": i} for i in range(20)]
    train1, _ = split_records(records, 0.20, seed=1)
    train2, _ = split_records(records, 0.20, seed=999)
    # Different seeds should produce different orderings (with high probability)
    assert train1 != train2


def test_split_small_dataset_has_at_least_one_val():
    records = [{"id": i} for i in range(3)]
    train, val = split_records(records, 0.10, seed=42)
    assert len(val) >= 1


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def test_load_jsonl_roundtrip():
    records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_path = f.name
    try:
        loaded = load_jsonl(tmp_path)
        assert loaded == records
    finally:
        os.unlink(tmp_path)


def test_load_jsonl_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": 1}\n\n{"id": 2}\n\n')
        tmp_path = f.name
    try:
        loaded = load_jsonl(tmp_path)
        assert len(loaded) == 2
    finally:
        os.unlink(tmp_path)


def test_load_jsonl_raises_on_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": 1}\nnot valid json\n')
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="invalid JSON"):
            load_jsonl(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_write_jsonl_creates_parent_dirs():
    records = [{"id": 1}]
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "nested", "dir", "output.jsonl")
        write_jsonl(records, output_path)
        assert os.path.exists(output_path)
        loaded = load_jsonl(output_path)
        assert loaded == records
