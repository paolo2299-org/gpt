"""Model assets stay external to disposable worktrees."""

from pathlib import Path

import pytest

from scripts import dev_app


def test_explicit_weights_path_is_resolved_from_worktree(tmp_path, monkeypatch):
    (tmp_path / "checkpoints").mkdir()
    monkeypatch.setenv("DEV_WEIGHTS_DIR", "checkpoints")
    assert dev_app.weights_directory(tmp_path) == tmp_path / "checkpoints"


def test_missing_explicit_weights_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_WEIGHTS_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="Weights directory not found"):
        dev_app.weights_directory(tmp_path)


def test_empty_weights_allow_page_development(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(dev_app, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dev_app, "make_server", lambda _host, _port, app, **_kwargs: app)
    app, assets = dev_app.create_server(Path(__file__).resolve().parents[1], "127.0.0.1")
    assert app.test_client().get("/").status_code == 200
    assert app.extensions["demos"].available() == []
    assert assets["weights_directory"] == str(tmp_path)
