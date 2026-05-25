from __future__ import annotations

import pytest

from app import create_app


class FakeCompleter:
    def complete(self, prompt, max_new_tokens, temperature, top_k, seed):
        return f"{prompt} continued with {max_new_tokens} tokens"


@pytest.fixture
def app():
    return create_app(
        {
            "TESTING": True,
            "LOAD_MODEL": False,
            "COMPLETER": FakeCompleter(),
            "MODEL_WEIGHTS_PATH": "test-model.pth",
            "MODEL_PRESET": "124M",
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()

