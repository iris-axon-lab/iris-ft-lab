# iris-ft-lab — developer shortcuts
# Source of truth is always in scripts/ and configs/; this file is a convenience layer.
# Run `make help` to see available targets.
#
# ── Canonical v2 workflow ──────────────────────────────────────────────────────
#   make generate-data   →  data/sft_train_100.jsonl          (100 examples, 25/tier)
#   make prepare-data    →  data/processed/sft/{train,valid}.jsonl  (90/10 split, seed=42)
#   make sft             →  outputs/sft_qwen25_3b_v2/          (LoRA adapter)
#   make eval            →  eval/results_raw.json + eval/results.md
#
# ── Legacy v1 targets (kept for reference, not the published results) ──────────
#   make sft-v1          →  outputs/sft_qwen25_3b/   (3-example scaffold, deprecated)
#   make eval-v1         →  scripts/eval_extraction.py with v1 adapter
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help verify generate-data prepare-data sft eval \
        sft-v1 eval-v1 eval-tracking dpo inference test validate-results \
        install-dpo-backend fuse-sft generate-dpo-data eval-dpo

help:
	@echo ""
	@echo "iris-ft-lab — canonical v2 targets:"
	@echo "  make generate-data     Generate data/sft_train_100.jsonl (100 examples, no model needed)"
	@echo "  make prepare-data      Split sft_train_100.jsonl → data/processed/sft/ (seed=42)"
	@echo "  make sft               Train LoRA adapter  →  outputs/sft_qwen25_3b_v2/"
	@echo "  make eval              Compare baseline vs fine-tuned  →  eval/results.md"
	@echo "  make validate-results  Check all expected artifacts exist and paths are consistent"
	@echo ""
	@echo "  make verify            Verify Python env, MLX, and Metal availability"
	@echo "  make inference         Print inference usage example"
	@echo "  make test              Run test suite"
	@echo ""
	@echo "DPO targets (v1 PROMOTED — eval/dpo_v1_results.md):"
	@echo "  make install-dpo-backend  Install mlx-lm-lora (separate from mlx-lm)"
	@echo "  make fuse-sft             Fuse SFT v2 into base (one-time, ~6 GB float16)"
	@echo "  make generate-dpo-data    Generate preference pairs (80 train + 12 valid)"
	@echo "  make dpo                  Train DPO LoRA  →  outputs/dpo_qwen25_3b_v1/"
	@echo "  make eval-dpo             Compare baseline vs DPO  →  eval/dpo_v1_results.md"
	@echo ""
	@echo "Legacy targets (v1 scaffold — not the published results):"
	@echo "  make sft-v1            Train with v1 config  →  outputs/sft_qwen25_3b/"
	@echo "  make eval-v1           Run eval_extraction.py with v1 adapter"
	@echo "  make eval-tracking     Run intention-reality tracking eval"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────

verify:
	python scripts/setup_verify.py

# ── Canonical v2 workflow ──────────────────────────────────────────────────────

# Step 1 — generate training data (no model required)
# Writes 100 balanced examples (25 per tier) to data/sft_train_100.jsonl
generate-data:
	python scripts/generate_synthetic_sft.py \
		--output data/sft_train_100.jsonl \
		--shuffle --seed 42

# Step 2 — validate and split the training data
# Reads data/sft_train_100.jsonl, writes data/processed/sft/{train,valid}.jsonl
prepare-data:
	python scripts/prepare_data.py \
		--sft data/sft_train_100.jsonl \
		--eval data/eval_gold.jsonl \
		--seed 42

# Step 3 — run SFT training
# Config: configs/sft_trace_qwen25_3b_v2.yaml
# Adapter output: outputs/sft_qwen25_3b_v2/
sft:
	python scripts/train_sft.py --config configs/sft_trace_qwen25_3b_v2.yaml

# Step 4 — run baseline vs fine-tuned comparison eval
# Writes eval/results_raw.json and eval/results.md
eval:
	python eval/run_eval.py \
		--eval-data data/eval_gold.jsonl \
		--config configs/sft_trace_qwen25_3b_v2.yaml \
		--ft-adapter outputs/sft_qwen25_3b_v2 \
		--output eval/results_raw.json \
		--write-md eval/results.md

# ── Validation ────────────────────────────────────────────────────────────────

# Verify that all expected artifacts exist and key paths are internally consistent.
# Runs without a model — checks files and config only.
validate-results:
	@echo ""
	@echo "Checking canonical v2 artifacts..."
	@test -f data/sft_train_100.jsonl      && echo "  [OK] data/sft_train_100.jsonl" \
	                                        || echo "  [MISSING] data/sft_train_100.jsonl  — run: make generate-data"
	@test -f data/eval_gold.jsonl          && echo "  [OK] data/eval_gold.jsonl" \
	                                        || echo "  [MISSING] data/eval_gold.jsonl"
	@test -f configs/sft_trace_qwen25_3b_v2.yaml \
	                                       && echo "  [OK] configs/sft_trace_qwen25_3b_v2.yaml" \
	                                        || echo "  [MISSING] configs/sft_trace_qwen25_3b_v2.yaml"
	@test -f eval/run_eval.py              && echo "  [OK] eval/run_eval.py" \
	                                        || echo "  [MISSING] eval/run_eval.py"
	@test -f eval/results.md              && echo "  [OK] eval/results.md" \
	                                        || echo "  [MISSING] eval/results.md  — run: make eval"
	@test -f eval/results_raw.json        && echo "  [OK] eval/results_raw.json" \
	                                        || echo "  [MISSING] eval/results_raw.json  — run: make eval"
	@test -f eval/notes.md                && echo "  [OK] eval/notes.md" \
	                                        || echo "  [MISSING] eval/notes.md"
	@echo ""
	@echo "Checking config → data path consistency..."
	@python3 -c "\
import yaml, sys; \
c = yaml.safe_load(open('configs/sft_trace_qwen25_3b_v2.yaml')); \
train = c['data']['train']; \
expect = 'data/processed/sft/train.jsonl'; \
ok = train == expect; \
print('  [OK] config data.train =', train) if ok else print('  [MISMATCH] config data.train =', train, '(expected', expect + ')'); \
sys.exit(0 if ok else 1)"
	@python3 -c "\
import yaml, sys; \
c = yaml.safe_load(open('configs/sft_trace_qwen25_3b_v2.yaml')); \
adapter = c['output']['adapter_path']; \
expect = 'outputs/sft_qwen25_3b_v2'; \
ok = adapter == expect; \
print('  [OK] config output.adapter_path =', adapter) if ok else print('  [MISMATCH] config output.adapter_path =', adapter, '(expected', expect + ')'); \
sys.exit(0 if ok else 1)"
	@echo ""
	@echo "Checking training data record count..."
	@python3 -c "\
import json, sys; \
lines = [l for l in open('data/sft_train_100.jsonl') if l.strip()]; \
n = len(lines); \
ok = n == 100; \
print('  [OK] sft_train_100.jsonl has', n, 'records') if ok else print('  [WARN] sft_train_100.jsonl has', n, 'records (expected 100)')"
	@echo ""
	@echo "Checking eval results claim accuracy..."
	@python3 -c "\
import json, sys; \
data = json.load(open('eval/results_raw.json')); \
ft = data.get('ft', {}); \
m = ft.get('metrics', {}); \
acc = m.get('accuracy', -1); \
correct = m.get('correct', -1); \
total = m.get('total', -1); \
print(f'  [OK] eval/results_raw.json: ft accuracy = {correct}/{total} ({100*acc:.1f}%)') \
  if abs(acc - 0.75) < 0.01 else \
  print(f'  [WARN] eval/results_raw.json: ft accuracy = {correct}/{total} ({100*acc:.1f}%) — README states 9/12 (75%)')"
	@echo ""
	@echo "Done. Re-run 'make eval' to regenerate results_raw.json and results.md."
	@echo ""

# ── Legacy v1 (scaffold — kept for reference) ─────────────────────────────────

# Original 3-example scaffold training — deprecated, not the published results
sft-v1:
	python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml

# Original extraction eval against v1 adapter
eval-v1:
	python scripts/eval_extraction.py \
		--eval-data data/eval_gold.jsonl \
		--config configs/sft_trace_qwen25_3b.yaml \
		--adapter outputs/sft_qwen25_3b

# ── DPO workflow (v1 PROMOTED — see eval/dpo_v1_results.md) ──────────────────

# Step 1 — install DPO backend (one-time)
install-dpo-backend:
	pip install -U mlx-lm-lora

# Step 2 — fuse SFT v2 adapter into base model (one-time, ~6 GB float16)
fuse-sft:
	python scripts/fuse_sft.py

# Step 3 — generate preference pairs (deterministic, ~80 train + 12 valid pairs)
generate-dpo-data:
	python scripts/generate_synthetic_dpo.py --n 80 --seed 42 --output data/processed/dpo/train.jsonl
	python scripts/generate_synthetic_dpo.py --n 12 --seed 99 --output data/processed/dpo/valid.jsonl
	python scripts/validate_dpo_data.py data/processed/dpo/train.jsonl
	python scripts/validate_dpo_data.py data/processed/dpo/valid.jsonl

# Step 4 — train DPO (requires fused-SFT model + preference pairs)
dpo:
	python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml

# Step 5 — evaluate DPO adapter against the 12-case gold set
eval-dpo:
	python scripts/eval_extraction.py \
		--eval-data data/eval_gold.jsonl \
		--config configs/dpo_trace_qwen25_3b.yaml \
		--adapter outputs/dpo_qwen25_3b_v1 \
		--verbose

eval-tracking:
	python scripts/eval_tracking.py

# ── Inference ─────────────────────────────────────────────────────────────────

inference:
	@echo ""
	@echo "Trace-style inference (pipe-friendly):"
	@echo ""
	@echo "  echo 'your trace text here' | python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b_v2.yaml \\"
	@echo "      --adapter outputs/sft_qwen25_3b_v2"
	@echo ""
	@echo "  python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b_v2.yaml \\"
	@echo "      --adapter outputs/sft_qwen25_3b_v2 \\"
	@echo "      --input 'Finished the retrospective write-up and sent it to the team.'"
	@echo ""
	@echo "  # Pipe output to jq for pretty-printing:"
	@echo "  echo '...' | python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b_v2.yaml \\"
	@echo "      --adapter outputs/sft_qwen25_3b_v2 | jq ."
	@echo ""

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v
