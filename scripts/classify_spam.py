from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken
import torch

from llm.classification import ID_TO_LABEL, classify_text, configure_classifier_model
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a text message with a chapter 6 fine-tuned model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default=None)
    parser.add_argument("--prompt", required=True, help="Message text to classify.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser()
    metadata = load_metadata(weights_path)
    preset = args.preset or metadata.get("preset") or DEFAULT_PRESET
    config = get_model_config(preset)
    max_length = args.max_length or metadata.get("max_length") or config["context_length"]
    labels = metadata.get("labels") or {str(key): value for key, value in ID_TO_LABEL.items()}
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(config)
    configure_classifier_model(
        model,
        emb_dim=config["emb_dim"],
        num_classes=int(metadata.get("num_classes", 2)),
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    predicted_id, probabilities = classify_text(
        model=model,
        text=args.prompt,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    predicted_label = labels.get(str(predicted_id), str(predicted_id))

    print("Task: spam classification")
    print("Preset:", preset)
    print("Device:", device)
    print("Input text:", args.prompt)
    print("Predicted label:", predicted_label)
    print("Probabilities:")
    for label_id, probability in enumerate(probabilities.tolist()):
        print(f"  {labels.get(str(label_id), str(label_id))}: {probability:.4f}")


if __name__ == "__main__":
    main()
