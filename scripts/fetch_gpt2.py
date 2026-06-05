"""Download OpenAI's GPT-2 open weights and save them as a demo checkpoint.

One-off, dev-only tool (book ch5, "alternative weight loading"). Sebastian Raschka
re-published OpenAI's GPT-2 weights as PyTorch state dicts whose keys already match
this project's :class:`llm.model.GPTModel`, so loading them needs only torch +
requests — no TensorFlow. The script downloads the state dict, loads it into a
``GPTModel`` to confirm it matches the preset, and re-saves a canonical
``weights/<preset>.pth`` (+ a ``.json`` sidecar, same shape ``scripts/pretrain.py``
writes) that ``scripts/generate.py`` and the web app load like any other checkpoint.

    .venv/bin/python scripts/fetch_gpt2.py --size 124M

writes ``weights/gpt2-small.pth`` (+ ``weights/gpt2-small.pth.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.config import get_model_config
from llm.model import GPTModel

HF_REPO_URL = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main"

# OpenAI size -> (this project's config preset, file name in the HF repo).
SIZE_TO_MODEL = {
    "124M": ("gpt2-small", "gpt2-small-124M.pth"),
    "355M": ("gpt2-medium", "gpt2-medium-355M.pth"),
    "774M": ("gpt2-large", "gpt2-large-774M.pth"),
    "1558M": ("gpt2-xl", "gpt2-xl-1558M.pth"),
}


def download_file(url: str, destination: Path) -> None:
    import requests
    from tqdm import tqdm

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))
    if destination.exists() and total and destination.stat().st_size == total:
        print(f"Up-to-date: {destination}")
        return
    with tqdm(total=total, unit="iB", unit_scale=True, desc=destination.name) as bar:
        with open(destination, "wb") as out:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=list(SIZE_TO_MODEL), default="124M")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to weights/<preset>.pth.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("weights/.gpt2-downloads"),
        help="Where to cache the downloaded state dict.",
    )
    return parser.parse_args()


def save_metadata(path: Path, metadata: dict) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    preset, hf_filename = SIZE_TO_MODEL[args.size]
    config = get_model_config(preset)
    output_path = (args.output or Path("weights") / f"{preset}.pth").expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    download_path = args.cache_dir / hf_filename
    print(f"Downloading GPT-2 {args.size} weights (preset '{preset}')...")
    download_file(f"{HF_REPO_URL}/{hf_filename}", download_path)

    # Load into our GPTModel to confirm the keys match the preset, then re-save a
    # canonical checkpoint. The ``mask`` buffers are constant and rebuilt at init,
    # so they're allowed to be absent from the downloaded state dict.
    state_dict = torch.load(download_path, map_location="cpu", weights_only=True)
    model = GPTModel(config)
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.endswith("mask")]
    if unexpected or missing:
        raise SystemExit(
            f"Downloaded weights do not match preset '{preset}'.\n"
            f"  missing keys:    {missing}\n"
            f"  unexpected keys: {unexpected}"
        )
    model.eval()

    torch.save(model.state_dict(), output_path)
    save_metadata(
        output_path,
        {
            "preset": preset,
            "model_config": config,
            "source": f"{HF_REPO_URL}/{hf_filename}",
            "model_size": args.size,
        },
    )

    print("Saved weights:", output_path)
    print("Saved metadata:", output_path.with_suffix(output_path.suffix + ".json"))


if __name__ == "__main__":
    main()
