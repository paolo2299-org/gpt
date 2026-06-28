from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.config import DEFAULT_PRESET, MODEL_PRESETS, get_model_config, get_training_settings
from llm.model import GPTModel
from llm.training import (
    create_train_val_loaders,
    evaluate_model,
    read_text,
    train_text_model,
)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_metadata(weights_path: Path) -> dict:
    metadata_path = weights_path.with_suffix(weights_path.suffix + ".json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def save_metadata(path: Path, metadata: dict) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def infer_base_preset(base_weights: Path, metadata: dict) -> str:
    if metadata.get("preset"):
        return metadata["preset"]
    if base_weights.name == "gpt2-small.pth":
        return "gpt2-small"
    return DEFAULT_PRESET


def default_best_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best.pth")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue language-model fine-tuning from a GPT checkpoint on plain text."
    )
    parser.add_argument("--base-weights", type=Path, default=Path("weights/gpt2-small.pth"))
    parser.add_argument("--preset", default=None, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--input-file", type=Path, default=Path("texts/jane-austen.cleaned.txt"))
    parser.add_argument("--output", type=Path, default=Path("weights/jane-austen-gpt2-small.pth"))
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help="Where to save the checkpoint with the lowest validation loss.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-epochs", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=positive_int, default=50)
    parser.add_argument("--eval-iter", type=positive_int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--start-context", default="It is a truth universally")
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Stop after this many optimizer steps. Useful for smoke tests.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Train from random weights if --base-weights is missing. Useful only for tiny smoke tests.",
    )
    return parser.parse_args()


def validate_split(train_loader, val_loader) -> None:
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader is empty. Use a smaller --batch-size, a shorter-context preset, "
            "or provide more text."
        )
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty. Provide more text or lower --train-ratio.")


def main() -> None:
    args = parse_args()
    base_weights = args.base_weights.expanduser()
    base_metadata = load_metadata(base_weights) if base_weights.exists() else {}
    preset = args.preset or infer_base_preset(base_weights, base_metadata)
    config = get_model_config(preset)
    settings = get_training_settings(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        train_ratio=args.train_ratio,
        start_context=args.start_context,
        max_steps=args.max_steps,
    )
    device = resolve_device(args.device)

    data_path = args.input_file.expanduser()
    text_data = read_text(data_path)
    train_loader, val_loader = create_train_val_loaders(text_data, config, settings)
    validate_split(train_loader, val_loader)

    torch.manual_seed(args.seed)
    model = GPTModel(config)
    if base_weights.exists():
        state_dict = torch.load(base_weights, map_location=device, weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [key for key in result.missing_keys if not key.endswith("mask")]
        if unexpected or missing:
            raise SystemExit(
                f"Base weights do not match preset '{preset}'.\n"
                f"  missing keys:    {missing}\n"
                f"  unexpected keys: {unexpected}"
            )
    elif not args.allow_random_init:
        raise FileNotFoundError(
            f"Base weights not found: {base_weights}. Fetch GPT-2 first with "
            f"'.venv/bin/python scripts/fetch_gpt2.py --size 124M', or pass --allow-random-init."
        )
    model.to(device)

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

    print("Task: language-model fine-tuning")
    print("Preset:", preset)
    print("Device:", device)
    print("Base weights:", base_weights if base_weights.exists() else "random initialization")
    print("Input file:", data_path)
    print("Output weights:", output_path)
    print("Best weights:", best_output_path)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    initial_train_loss, initial_val_loss = evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        eval_iter=args.eval_iter,
    )
    print(f"Initial train loss: {initial_train_loss:.3f}")
    print(f"Initial val loss: {initial_val_loss:.3f}")

    train_losses, val_losses, tokens_seen, model = train_text_model(
        model=model,
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

    metadata = {
        "task": "language-model-finetuning",
        "preset": preset,
        "seed": args.seed,
        "model_config": config,
        "base_weights": str(base_weights) if base_weights.exists() else None,
        "base_metadata": base_metadata,
        "input_file": str(data_path),
        "training_settings": settings,
        "data": {
            "characters": len(text_data),
            "training_batches": len(train_loader),
            "validation_batches": len(val_loader),
        },
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "best_checkpoint": best_metadata,
    }
    save_metadata(output_path, metadata)
    if best_metadata is not None:
        save_metadata(best_output_path, {**metadata, "checkpoint": "best_validation"})
    else:
        print("No finite validation loss was observed; best checkpoint was not saved.")

    print("Saved weights:", output_path)
    print("Saved metadata:", output_path.with_suffix(output_path.suffix + ".json"))
    if best_metadata is not None:
        print("Saved best weights:", best_output_path)
        print("Saved best metadata:", best_output_path.with_suffix(best_output_path.suffix + ".json"))


if __name__ == "__main__":
    main()
