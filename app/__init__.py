"""Flask application factory for the multi-demo LLM web UI."""

from __future__ import annotations

import os

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app(test_config: dict[str, object] | None = None) -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]

    app.config.from_object("app.config.Config")
    app.config.update(
        WEIGHTS_DIR=os.environ.get("WEIGHTS_DIR", app.config["WEIGHTS_DIR"]),
        MODEL_DEVICE=os.environ.get("MODEL_DEVICE", app.config["MODEL_DEVICE"]),
        SITE_TITLE=os.environ.get("SITE_TITLE", app.config["SITE_TITLE"]),
    )

    if test_config:
        app.config.update(test_config)

    _init_registry(app)

    from app.routes.pages import bp as pages_bp

    app.register_blueprint(pages_bp)
    return app


def _init_registry(app: Flask) -> None:
    injected_registry = app.config.get("DEMO_REGISTRY")
    if injected_registry is not None:
        app.extensions["demos"] = injected_registry
        return

    from app.services.registry import DemoRegistry

    app.extensions["demos"] = DemoRegistry(
        weights_dir=app.config["WEIGHTS_DIR"],
        device_name=app.config["MODEL_DEVICE"],
        runners=app.config.get("DEMO_RUNNERS") or None,
    )
