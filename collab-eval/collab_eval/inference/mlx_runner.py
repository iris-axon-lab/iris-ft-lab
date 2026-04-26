"""
MLX-LM inference runner.

Wraps mlx_lm.load and mlx_lm.generate with graceful degradation:
- If mlx_lm is not installed, returns (False, reason) from check_mlx_available.
- If the model cannot be fetched or loaded, raises RuntimeError with a
  human-readable message that the caller should surface to the user.

Default model: mlx-community/Qwen2.5-3B-Instruct-4bit
No other default model is defined here.

All inference paths are optional. The rest of collab_eval runs offline
without this module.
"""

from __future__ import annotations

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def check_mlx_available() -> tuple[bool, str]:
    """
    Return (available, reason_if_not).

    Does not attempt to load a model; only checks that mlx_lm is importable.
    """
    try:
        import mlx_lm  # noqa: F401
        return True, ""
    except ImportError:
        return False, "mlx_lm not installed (pip install mlx-lm)"


def load_model(model_id: str):
    """
    Load model and tokenizer via mlx_lm.load.

    Returns (model, tokenizer) on success.
    Raises RuntimeError with a human-readable message on any failure
    (mlx_lm not installed, model not found, download failure, etc.).
    """
    avail, reason = check_mlx_available()
    if not avail:
        raise RuntimeError(reason)
    try:
        from mlx_lm import load
        model, tokenizer = load(model_id)
        return model, tokenizer
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {model_id!r}: {exc}") from exc


def generate(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """
    Run MLX inference and return the generated completion text.

    Caller is responsible for ensuring model and tokenizer are already loaded
    (via load_model). This function does not handle model-unavailable gracefully
    because by the time it is called, the model is already in memory.
    """
    from mlx_lm import generate as _mlx_generate
    return _mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
