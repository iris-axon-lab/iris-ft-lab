"""
Tests for SFT data schema validation.

Validates every record in data/sft_collab_eval_sample.jsonl and
validates SFT records built from small in-memory fixtures.

All tests run offline without model, GPU, or API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SAMPLE_FILE = _ROOT / "data" / "sft_collab_eval_sample.jsonl"

_VALID_DIMENSIONS = {
    "data_preservation", "unit_consistency", "format_validity", "completeness"
}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_REQUIRED_METADATA_KEYS = {
    "task_type", "difficulty", "primary_dimension", "generation_seed", "case_id"
}
_FORBIDDEN_STRINGS = [
    "microsoft", "sharepoint", "azure", "office 365",
    "internal", "confidential", "proprietary",
]


def _load_sample() -> list[dict]:
    assert _SAMPLE_FILE.exists(), (
        f"Sample file not found: {_SAMPLE_FILE}\n"
        "Generate with: python -c \"import json, sys; sys.path.insert(0, '.'); "
        "from scripts.build_sft_data import build_sft_records_from_generator; "
        "records = build_sft_records_from_generator(); "
        "[open('data/sft_collab_eval_sample.jsonl','w').write(json.dumps(r)+'\\n') for r in records]\""
    )
    records = []
    with _SAMPLE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_fixture_records() -> list[dict]:
    from collab_eval.generation.spreadsheet_generator import generate_cases
    from scripts.build_sft_data import case_to_sft_record
    import dataclasses
    cases = generate_cases(n=6, seed=555)
    return [case_to_sft_record(dataclasses.asdict(c)) for c in cases]


class TestSampleFileExists:
    def test_sample_file_present(self):
        assert _SAMPLE_FILE.exists(), f"Missing committed sample: {_SAMPLE_FILE}"

    def test_sample_has_expected_count(self):
        records = _load_sample()
        assert 1 <= len(records) <= 20, f"Expected 1-20 records, got {len(records)}"


class TestSFTRecordSchema:
    @pytest.fixture(params=["sample", "fixture"])
    def records(self, request):
        if request.param == "sample":
            return _load_sample()
        else:
            return _build_fixture_records()

    def test_messages_field_present(self, records):
        for r in records:
            assert "messages" in r, f"Missing 'messages' in record"

    def test_messages_has_three_elements(self, records):
        for r in records:
            msgs = r["messages"]
            assert len(msgs) == 3, (
                f"Expected 3 messages [system, user, assistant], got {len(msgs)}"
            )

    def test_messages_roles_in_order(self, records):
        for r in records:
            roles = [m["role"] for m in r["messages"]]
            assert roles == ["system", "user", "assistant"], (
                f"Expected [system, user, assistant], got {roles}"
            )

    def test_assistant_output_present(self, records):
        for r in records:
            assistant_content = r["messages"][2]["content"]
            assert assistant_content and assistant_content.strip(), (
                "Assistant message content is empty"
            )

    def test_metadata_present(self, records):
        for r in records:
            assert "metadata" in r, "Missing 'metadata' key"

    def test_all_metadata_keys_present(self, records):
        for r in records:
            meta = r["metadata"]
            missing = _REQUIRED_METADATA_KEYS - meta.keys()
            assert missing == set(), f"Missing metadata keys: {missing}"

    def test_metadata_task_type(self, records):
        for r in records:
            assert r["metadata"]["task_type"] == "spreadsheet_clean"

    def test_metadata_difficulty_valid(self, records):
        for r in records:
            diff = r["metadata"]["difficulty"]
            assert diff in _VALID_DIFFICULTIES, (
                f"Invalid difficulty '{diff}' (must be one of {_VALID_DIFFICULTIES})"
            )

    def test_metadata_primary_dimension_valid(self, records):
        for r in records:
            dim = r["metadata"]["primary_dimension"]
            assert dim in _VALID_DIMENSIONS, (
                f"Invalid primary_dimension '{dim}'"
            )

    def test_metadata_case_id_present(self, records):
        for r in records:
            assert r["metadata"]["case_id"], "case_id must be non-empty"

    def test_metadata_generation_seed_is_int(self, records):
        for r in records:
            seed = r["metadata"]["generation_seed"]
            assert isinstance(seed, int), f"generation_seed must be int, got {type(seed)}"


class TestGoldOutputDeterminism:
    def test_gold_output_is_parseable_csv(self, tmp_path):
        from collab_eval.graders import deterministic as det
        records = _load_sample()
        for r in records:
            gold = r["messages"][2]["content"]
            assert det.csv_parseable(gold), (
                f"{r['metadata']['case_id']}: gold output is not parseable CSV"
            )

    def test_gold_output_deterministic_from_seed(self):
        """Rebuilding from the same seed produces identical assistant content."""
        from collab_eval.generation.spreadsheet_generator import generate_cases
        from scripts.build_sft_data import case_to_sft_record
        import dataclasses
        cases_a = generate_cases(n=5, seed=999)
        cases_b = generate_cases(n=5, seed=999)
        records_a = [case_to_sft_record(dataclasses.asdict(c)) for c in cases_a]
        records_b = [case_to_sft_record(dataclasses.asdict(c)) for c in cases_b]
        for a, b in zip(records_a, records_b):
            assert a["messages"][2]["content"] == b["messages"][2]["content"], (
                "Gold output is not deterministic from seed"
            )
            assert a["metadata"]["case_id"] == b["metadata"]["case_id"]


class TestNoPrivateData:
    def test_no_forbidden_strings_in_sample(self):
        records = _load_sample()
        for r in records:
            content = json.dumps(r).lower()
            for forbidden in _FORBIDDEN_STRINGS:
                assert forbidden not in content, (
                    f"{r['metadata']['case_id']}: contains forbidden string '{forbidden}'"
                )

    def test_no_model_generated_content(self):
        """Gold outputs must come from the generator, not an LLM.
        We verify this indirectly: rebuilding from the exact same (n, seed) as the
        sample file produces identical content. The committed sample uses n=20, seed=999.
        """
        from collab_eval.generation.spreadsheet_generator import generate_cases
        from scripts.build_sft_data import case_to_sft_record
        import dataclasses
        # Must match the n used to generate the sample file (n=20, seed=999)
        cases = generate_cases(n=20, seed=999)
        fresh_records = [case_to_sft_record(dataclasses.asdict(c)) for c in cases]
        # Compare against the committed sample (all 20 records)
        sample = _load_sample()
        for fresh, committed in zip(fresh_records, sample):
            assert fresh["messages"][2]["content"] == committed["messages"][2]["content"], (
                "Sample gold output does not match deterministic generator output — "
                "possible non-deterministic (e.g. model-generated) content"
            )
