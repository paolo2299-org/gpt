"""Flask application factory for the local LLM completion UI."""

from __future__ import annotations

import os

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app(test_config: dict[str, object] | None = None) -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]

    app.config.from_object("app.config.Config")
    app.config.update(
        MODEL_WEIGHTS_PATH=os.environ.get("MODEL_WEIGHTS_PATH"),
        MODEL_PRESET=os.environ.get("MODEL_PRESET", app.config["MODEL_PRESET"]),
        MODEL_DEVICE=os.environ.get("MODEL_DEVICE", app.config["MODEL_DEVICE"]),
        SITE_TITLE=os.environ.get("SITE_TITLE", app.config["SITE_TITLE"]),
        AUTHOR_NAME=os.environ.get("AUTHOR_NAME", app.config["AUTHOR_NAME"]),
    )

    if test_config:
        app.config.update(test_config)

    _init_completer(app)

    from app.routes.pages import bp as pages_bp

    app.register_blueprint(pages_bp)
    return app


def _init_completer(app: Flask) -> None:
    injected_completer = app.config.get("COMPLETER")
    if injected_completer is not None:
        app.extensions["llm_completer"] = injected_completer
        return

    if not app.config.get("LOAD_MODEL", True):
        return

    if not app.config.get("MODEL_WEIGHTS_PATH"):
        raise RuntimeError("MODEL_WEIGHTS_PATH environment variable is required")

    from app.services.completer import LLMCompleter

    app.extensions["llm_completer"] = LLMCompleter.from_config(app.config)

