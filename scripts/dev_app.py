"""Load shared model checkpoints for inference without copying or modifying them."""

import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from werkzeug.serving import make_server


def weights_directory(root: Path) -> Path:
    explicit = os.environ.get("DEV_WEIGHTS_DIR") or os.environ.get("WEIGHTS_DIR")
    if explicit:
        path = Path(explicit).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_dir():
            raise ValueError(f"Weights directory not found: {path}. Set DEV_WEIGHTS_DIR to a native host path.")
        return path
    common = Path(subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True,
    ).strip())
    candidates = (root / "weights", common.parent / "weights", Path("/srv/gpt/weights"))
    return next((path for path in candidates if any(path.glob("*.pth"))), root / "weights")


def create_server(root: Path, host: str):
    load_dotenv(root / ".env", override=False)
    weights = weights_directory(root)
    os.environ["WEIGHTS_DIR"] = str(weights)
    os.environ.setdefault("MODEL_DEVICE", "auto")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from app import create_app

    if not any(weights.glob("*.pth")):
        print(f"No checkpoints in {weights}; the landing page will have no model demos.", flush=True)
    return make_server(host, 0, create_app(), threaded=True), {"weights_directory": str(weights)}
