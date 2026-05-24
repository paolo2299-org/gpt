"""Flask application configuration."""

from __future__ import annotations

from pathlib import Path


class Config:
    SECRET_KEY = "dev-secret-key"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False

    MODEL_WEIGHTS_PATH = str(Path("model.jane-austen-5.pth"))
    MODEL_PRESET = "book-124M"
    MODEL_DEVICE = "auto"
    LOAD_MODEL = True

    DEFAULT_MAX_NEW_TOKENS = 80
    MAX_NEW_TOKENS_LIMIT = 200
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_TOP_K = 50
    DEFAULT_SEED = 123

