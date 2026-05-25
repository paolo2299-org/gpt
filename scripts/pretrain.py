from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm_demo.config import (
    DEFAULT_PRESET,
    MODEL_PRESETS,
    get_model_config,
    get_training_settings,
)
from llm_demo.training import pretrain_from_text, read_text


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the demo GPT model on a text file.")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help="Where to save the checkpoint with the lowest validation loss.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-iter", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--start-context", default=None)
    return parser.parse_args()


def save_metadata(path: Path, metadata: dict) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def default_best_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best.pth")


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser()
    if not input_file.exists():
        raise FileNotFoundError(f"Input text file not found: {input_file}")

    config = get_model_config(args.preset)
    settings = get_training_settings(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        train_ratio=args.train_ratio,
        start_context=args.start_context,
    )
    device = resolve_device(args.device)
    text_data = read_text(input_file)
    output_path = args.output.expanduser()
    best_output_path = (args.best_output or default_best_output_path(output_path)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint = {
        "path": best_output_path,
        "epoch": None,
        "global_step": None,
        "train_loss": None,
        "val_loss": math.inf,
        "tokens_seen": None,
    }

    def save_best_checkpoint(model, epoch, global_step, train_loss, val_loss, tokens_seen):
        if not math.isfinite(val_loss) or val_loss >= best_checkpoint["val_loss"]:
            return

        torch.save(model.state_dict(), best_output_path)
        best_checkpoint.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_seen": tokens_seen,
            }
        )
        print(f"Saved new best checkpoint: {best_output_path} (Val loss {val_loss:.3f})")

    print("Preset:", args.preset)
    print("Device:", device)
    print("Input file:", input_file)
    print("Output weights:", output_path)
    print("Best weights:", best_output_path)

    train_losses, val_losses, tokens_seen, model = pretrain_from_text(
        text_data=text_data,
        gpt_config=config,
        settings=settings,
        device=device,
        seed=args.seed,
        on_eval=save_best_checkpoint,
    )

    torch.save(model.state_dict(), output_path)
    best_metadata = None
    if best_checkpoint["epoch"] is not None:
        best_metadata = {
            "path": str(best_checkpoint["path"]),
            "epoch": best_checkpoint["epoch"],
            "global_step": best_checkpoint["global_step"],
            "train_loss": best_checkpoint["train_loss"],
            "val_loss": best_checkpoint["val_loss"],
            "tokens_seen": best_checkpoint["tokens_seen"],
        }

    save_metadata(
        output_path,
        {
            "preset": args.preset,
            "seed": args.seed,
            "model_config": config,
            "training_settings": settings,
            "input_file": str(input_file),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "best_checkpoint": best_metadata,
        },
    )
    if best_metadata is not None:
        save_metadata(
            best_output_path,
            {
                "preset": args.preset,
                "seed": args.seed,
                "model_config": config,
                "training_settings": settings,
                "input_file": str(input_file),
                "checkpoint": "best_validation",
                "best_checkpoint": best_metadata,
            },
        )
    else:
        print("No finite validation loss was observed; best checkpoint was not saved.")

    print("Saved weights:", output_path)
    print("Saved metadata:", output_path.with_suffix(output_path.suffix + ".json"))
    if best_metadata is not None:
        print("Saved best weights:", best_output_path)
        print("Saved best metadata:", best_output_path.with_suffix(best_output_path.suffix + ".json"))


if __name__ == "__main__":
    main()
