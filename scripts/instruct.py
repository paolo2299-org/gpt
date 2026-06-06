from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken
import torch

from llm.config import DEFAULT_PRESET, MODEL_PRESETS, get_model_config
from llm.instruction import generate_instruction_response
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
    parser = argparse.ArgumentParser(description="Generate a response with a chapter 7 instruction-tuned model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default=None)
    parser.add_argument("--prompt", required=True, help="Instruction for the model.")
    parser.add_argument("--input", default="", help="Optional input block paired with the instruction.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser()
    metadata = load_metadata(weights_path)
    preset = args.preset or metadata.get("preset") or DEFAULT_PRESET
    config = get_model_config(preset)
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(config)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    response = generate_instruction_response(
        model=model,
        instruction=args.prompt,
        prompt_input=args.input,
        tokenizer=tokenizer,
        device=device,
        context_size=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("Task: instruction following")
    print("Preset:", preset)
    print("Seed:", args.seed)
    print("Device:", device)
    print("Instruction:", args.prompt)
    if args.input:
        print("Input:", args.input)
    print("Response:\n", response)


if __name__ == "__main__":
    main()
