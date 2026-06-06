from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.classification import (
    calc_accuracy_loader,
    configure_classifier_model,
    create_spam_dataloaders,
    ensure_spam_tsv,
    prepare_spam_data,
    train_classifier_simple,
)
from llm.config import DEFAULT_PRESET, MODEL_PRESETS, get_model_config
from llm.model import GPTModel


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT for the chapter 6 SMS spam classifier.")
    parser.add_argument("--base-weights", type=Path, default=Path("weights/gpt2-small.pth"))
    parser.add_argument("--preset", default=None, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--output", type=Path, default=Path("weights/spam-classifier.pth"))
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help="Where to save the checkpoint with the lowest validation loss.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sms-tsv",
        type=Path,
        default=None,
        help="Existing SMSSpamCollection.tsv path. If omitted, the script downloads the dataset under --data-dir.",
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/spam-prepared"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Train from random weights if --base-weights is missing. Useful only for tiny smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_weights = args.base_weights.expanduser()
    base_metadata = load_metadata(base_weights) if base_weights.exists() else {}
    preset = args.preset or infer_base_preset(base_weights, base_metadata)
    config = get_model_config(preset)
    device = resolve_device(args.device)

    sms_tsv_path = args.sms_tsv.expanduser() if args.sms_tsv else ensure_spam_tsv(args.data_dir)
    train_csv, val_csv, test_csv = prepare_spam_data(sms_tsv_path, args.prepared_dir.expanduser())
    train_loader, val_loader, test_loader, max_length = create_spam_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Training loader is empty. Use a smaller --batch-size than the prepared train set size "
            f"({len(train_loader.dataset)}), or provide more data."
        )
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation set is empty. Provide more data or adjust the split settings.")
    if max_length > config["context_length"]:
        raise ValueError(
            f"Dataset max length {max_length} exceeds context length {config['context_length']} "
            f"for preset '{preset}'."
        )

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

    configure_classifier_model(model, emb_dim=config["emb_dim"], num_classes=2)
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
        "examples_seen": None,
    }

    def save_best_checkpoint(model, epoch, global_step, train_loss, val_loss, examples_seen):
        if not math.isfinite(val_loss) or val_loss >= best_checkpoint["val_loss"]:
            return
        torch.save(model.state_dict(), best_output_path)
        best_checkpoint.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "examples_seen": examples_seen,
            }
        )
        print(f"Saved new best checkpoint: {best_output_path} (Val loss {val_loss:.3f})")

    print("Task: spam classification")
    print("Preset:", preset)
    print("Device:", device)
    print("Base weights:", base_weights if base_weights.exists() else "random initialization")
    print("Output weights:", output_path)
    print("Best weights:", best_output_path)
    print("Prepared data:", args.prepared_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        on_eval=save_best_checkpoint,
    )

    test_accuracy = calc_accuracy_loader(test_loader, model, device) if test_loader is not None else None
    if test_accuracy is not None:
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), output_path)
    best_metadata = None
    if best_checkpoint["epoch"] is not None:
        best_metadata = {
            "path": str(best_checkpoint["path"]),
            "epoch": best_checkpoint["epoch"],
            "global_step": best_checkpoint["global_step"],
            "train_loss": best_checkpoint["train_loss"],
            "val_loss": best_checkpoint["val_loss"],
            "examples_seen": best_checkpoint["examples_seen"],
        }

    metadata = {
        "task": "spam-classification",
        "preset": preset,
        "seed": args.seed,
        "model_config": config,
        "base_weights": str(base_weights) if base_weights.exists() else None,
        "num_classes": 2,
        "labels": {"0": "ham", "1": "spam"},
        "max_length": max_length,
        "training_settings": {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "eval_freq": args.eval_freq,
            "eval_iter": args.eval_iter,
        },
        "data": {
            "sms_tsv": str(sms_tsv_path),
            "train_csv": str(train_csv),
            "validation_csv": str(val_csv),
            "test_csv": str(test_csv),
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_accuracy": test_accuracy,
        "examples_seen": examples_seen,
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
