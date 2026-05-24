from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tiktoken
import torch

from llm_demo.config import get_model_config
from llm_demo.generation import generate, text_to_token_ids, token_ids_to_text
from llm_demo.model import GPTModel, create_dataloader_v1
from llm_demo.training import pretrain_from_text


TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class LLMDemoTests(unittest.TestCase):
    def test_demo_small_forward_shape(self):
        config = get_model_config("demo-small")
        model = GPTModel(config)
        input_ids = torch.randint(0, config["vocab_size"], (2, 4))

        logits = model(input_ids)

        self.assertEqual(logits.shape, (2, 4, config["vocab_size"]))

    def test_seeded_random_generation_is_deterministic(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        prompt = "Hello, I am"

        def run_once():
            torch.manual_seed(123)
            model = GPTModel(TINY_CONFIG)
            model.eval()
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(prompt, tokenizer),
                max_new_tokens=4,
                context_size=TINY_CONFIG["context_length"],
            )
            return token_ids_to_text(token_ids, tokenizer)

        self.assertEqual(run_once(), run_once())

    def test_create_dataloader_v1_from_text(self):
        text = "Every effort moves you forward. " * 20

        dataloader = create_dataloader_v1(
            text,
            batch_size=2,
            max_length=8,
            stride=8,
            shuffle=False,
            drop_last=False,
        )
        input_batch, target_batch = next(iter(dataloader))

        self.assertEqual(input_batch.shape, (2, 8))
        self.assertEqual(target_batch.shape, (2, 8))

    def test_tiny_pretraining_saves_and_loads_weights(self):
        settings = {
            "learning_rate": 5e-4,
            "num_epochs": 1,
            "batch_size": 2,
            "weight_decay": 0.1,
            "eval_freq": 1000,
            "eval_iter": 1,
            "train_ratio": 0.8,
            "start_context": "Every effort moves you",
        }
        text = "Every effort moves you forward. " * 80

        _, _, _, model = pretrain_from_text(
            text_data=text,
            gpt_config=TINY_CONFIG,
            settings=settings,
            device=torch.device("cpu"),
            seed=123,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "tiny-model.pth"
            torch.save(model.state_dict(), weights_path)

            loaded_model = GPTModel(TINY_CONFIG)
            loaded_model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
            loaded_model.eval()

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = generate(
            model=loaded_model,
            idx=text_to_token_ids("Every effort", tokenizer),
            max_new_tokens=2,
            context_size=TINY_CONFIG["context_length"],
        )

        self.assertEqual(token_ids.shape[1], 4)

    def test_pretraining_calls_eval_callback(self):
        settings = {
            "learning_rate": 5e-4,
            "num_epochs": 1,
            "batch_size": 2,
            "weight_decay": 0.1,
            "eval_freq": 1000,
            "eval_iter": 1,
            "train_ratio": 0.8,
            "start_context": "Every effort moves you",
        }
        evals = []

        def on_eval(**kwargs):
            evals.append(kwargs)

        pretrain_from_text(
            text_data="Every effort moves you forward. " * 80,
            gpt_config=TINY_CONFIG,
            settings=settings,
            device=torch.device("cpu"),
            seed=123,
            on_eval=on_eval,
        )

        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["epoch"], 1)
        self.assertEqual(evals[0]["global_step"], 0)
        self.assertIn("val_loss", evals[0])


if __name__ == "__main__":
    unittest.main()
