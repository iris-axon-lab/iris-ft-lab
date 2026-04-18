# iris-ft-lab — developer shortcuts
# Source of truth is always in scripts/; this Makefile is a convenience layer only.
# Run `make help` to see available targets.

.PHONY: help verify generate-data prepare-data sft dpo eval eval-base eval-tracking inference test

help:
	@echo ""
	@echo "iris-ft-lab targets:"
	@echo "  make verify          Verify Python env, MLX, and Metal availability"
	@echo "  make prepare-data    Validate and split sample data into data/processed/"
	@echo "  make sft             Run SFT training (reads configs/sft_trace_qwen25_3b.yaml)"
	@echo "  make dpo             Run DPO scaffold (see train_dpo.py for current status)"
	@echo "  make eval            Run extraction eval with the trained SFT adapter"
	@echo "  make eval-base       Run extraction eval against base model (no adapter)"
	@echo "  make eval-tracking   Run intention-reality tracking eval"
	@echo "  make inference       Print inference usage example"
	@echo "  make test            Run test suite"
	@echo ""

verify:
	python scripts/setup_verify.py

# Generate the extended synthetic SFT dataset (no model required)
generate-data:
	python scripts/generate_synthetic_sft.py \
		--output data/sample_sft_extended.jsonl \
		--shuffle --seed 42

prepare-data:
	python scripts/prepare_data.py \
		--sft data/sample_sft.jsonl \
		--dpo data/sample_dpo.jsonl \
		--eval data/eval_gold.jsonl \
		--seed 42

sft:
	python scripts/train_sft.py --config configs/sft_trace_qwen25_3b.yaml

dpo:
	python scripts/train_dpo.py --config configs/dpo_trace_qwen25_3b.yaml

# Eval against the trained SFT adapter (primary eval target)
eval:
	python scripts/eval_extraction.py \
		--eval-data data/eval_gold.jsonl \
		--config configs/sft_trace_qwen25_3b.yaml \
		--adapter outputs/sft_qwen25_3b

# Eval against the base model — use this as a baseline to measure adapter improvement
eval-base:
	python scripts/eval_extraction.py \
		--eval-data data/eval_gold.jsonl \
		--config configs/sft_trace_qwen25_3b.yaml

eval-tracking:
	python scripts/eval_tracking.py

inference:
	@echo ""
	@echo "Trace-style inference (pipe-friendly):"
	@echo ""
	@echo "  echo 'your trace text here' | python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b.yaml"
	@echo ""
	@echo "  python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b.yaml \\"
	@echo "      --input 'Finished the retrospective write-up and sent it to the team.'"
	@echo ""
	@echo "  # Pipe output to jq for pretty-printing:"
	@echo "  echo '...' | python scripts/run_trace_style_inference.py \\"
	@echo "      --config configs/sft_trace_qwen25_3b.yaml | jq ."
	@echo ""

test:
	pytest tests/ -v
