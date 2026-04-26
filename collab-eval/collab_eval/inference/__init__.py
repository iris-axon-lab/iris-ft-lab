"""
MLX-LM inference utilities for collab_eval.

Degrades gracefully when mlx_lm is not installed or a model is unavailable.
All inference calls are optional; the rest of the harness runs offline.
"""

from collab_eval.inference.mlx_runner import check_mlx_available, load_model, generate

__all__ = ["check_mlx_available", "load_model", "generate"]
