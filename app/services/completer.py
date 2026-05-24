"""Load the trained GPT model once and use it for prompt completion."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import tiktoken
import torch

from llm_demo.config import get_model_config
from llm_demo.generation import generate, text_to_token_ids, token_ids_to_text
from llm_demo.model import GPTModel


class LLMCompleter:
    def __init__(self, weights_path: Path, preset: str, device_name: str = "auto") -> None:
        self.weights_path = weights_path
        self.preset = preset
        print(
            f"Resolving model device from setting '{device_name}'...",
            flush=True,
        )
        self.device = resolve_device(device_name)
        print(
            f"Loading tokenizer and model config for preset '{preset}'...",
            flush=True,
        )
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.config = get_model_config(preset)
        self.lock = Lock()
        self.model = self._load_model()
        print(
            f"Model ready on {self.device}.",
            flush=True,
        )

    @classmethod
    def from_config(cls, config) -> "LLMCompleter":
        return cls(
            weights_path=Path(config["MODEL_WEIGHTS_PATH"]),
            preset=config["MODEL_PRESET"],
            device_name=config["MODEL_DEVICE"],
        )

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int,
    ) -> str:
        with self.lock:
            torch.manual_seed(seed)
            token_ids = generate(
                model=self.model,
                idx=text_to_token_ids(prompt, self.tokenizer).to(self.device),
                max_new_tokens=max_new_tokens,
                context_size=self.config["context_length"],
                temperature=temperature,
                top_k=top_k,
            )
        return token_ids_to_text(token_ids, self.tokenizer)

    def _load_model(self) -> GPTModel:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        print(
            f"Building model architecture for preset '{self.preset}'...",
            flush=True,
        )
        model = GPTModel(self.config)
        print(
            f"Loading model weights from {self.weights_path}...",
            flush=True,
        )
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        print("Applying model weights...", flush=True)
        model.load_state_dict(state_dict)
        print(f"Moving model to {self.device}...", flush=True)
        model.to(self.device)
        model.eval()
        return model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)
