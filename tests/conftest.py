from __future__ import annotations

import pytest

from app import create_app


class FakeCompleter:
    def complete(self, prompt, max_new_tokens, temperature, top_k, seed):
        return f"{prompt} continued with {max_new_tokens} tokens"


@pytest.fixture
def weights_dir(tmp_path):
    return tmp_path


@pytest.fixture
def make_client(weights_dir):
    """Build a test client against a temp weights dir.

    ``runners`` injects fake runners (keyed by demo slug) so a demo is available
    without real weights. ``files`` creates placeholder weight files so a demo is
    available via file presence (landing page only — no model is loaded).
    """

    def _make(runners: dict | None = None, files=()):
        for name in files:
            (weights_dir / name).write_bytes(b"")
        app = create_app(
            {
                "TESTING": True,
                "WEIGHTS_DIR": str(weights_dir),
                "DEMO_RUNNERS": runners if runners is not None else {"jane": FakeCompleter()},
            }
        )
        return app.test_client()

    return _make


@pytest.fixture
def client(make_client):
    return make_client()
