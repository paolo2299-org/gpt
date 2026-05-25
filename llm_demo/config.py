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
    "book-124M": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
}

DEFAULT_PRESET = "demo-small"

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

