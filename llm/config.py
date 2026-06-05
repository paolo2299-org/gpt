from __future__ import annotations

from copy import deepcopy
from pathlib import Path


MODEL_PRESETS = {
    "demo-small": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    "124M": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    # OpenAI GPT-2 open weights (book ch5). Same GPTModel architecture as the
    # from-scratch presets above, but with the config the released weights were
    # trained with: full 1024-token context, query/key/value biases, no dropout.
    # Fetch the real weights into this shape with ``scripts/fetch_gpt2.py``.
    "gpt2-small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
}

DEFAULT_PRESET = "124M"

DEFAULT_TRAINING_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 1,
    "train_ratio": 0.90,
    "start_context": "I wandered lonely as a",
}

BOOK_TEXT_PATH = (
    Path.home()
    / "projects"
    / "LLMs-from-scratch"
    / "ch02"
    / "01_main-chapter-code"
    / "the-verdict.txt"
)


def get_model_config(preset: str = DEFAULT_PRESET) -> dict:
    try:
        return deepcopy(MODEL_PRESETS[preset])
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Choose one of: {choices}") from exc


def get_training_settings(**overrides) -> dict:
    settings = deepcopy(DEFAULT_TRAINING_SETTINGS)
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings

