"""Flask application configuration."""

from __future__ import annotations

from pathlib import Path


class Config:
    SECRET_KEY = "dev-secret-key"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False

    SITE_TITLE = "LLM From Scratch Demos"

    # Directory holding the demos' weight files (one .pth per demo, named by each
    # demo's ``weights_filename``). A demo is shown only if its file is present.
    WEIGHTS_DIR = "weights"
    MODEL_DEVICE = "auto"

    DEFAULT_MAX_NEW_TOKENS = 80
    MAX_NEW_TOKENS_LIMIT = 200
    # Sample rather than decode greedily: temperature 0.0 (argmax) degenerates into
    # repetitive loops on GPT-2-class models. A fixed seed keeps results reproducible.
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 50
    DEFAULT_SEED = 123

