from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken
import torch

from llm_demo.config import DEFAULT_PRESET, MODEL_PRESETS, get_model_config
from llm_demo.generation import generate, text_to_token_ids, token_ids_to_text
from llm_demo.model import GPTModel


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an untrained GPT model and generate text.")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="Hello, I am")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = get_model_config(args.preset)

    torch.manual_seed(args.seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(config)
    model.to(device)
    model.eval()

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=config["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("Preset:", args.preset)
    print("Seed:", args.seed)
    print("Device:", device)
    print("Input text:", args.prompt)
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()

