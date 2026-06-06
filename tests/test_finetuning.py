from __future__ import annotations

import json

import pandas as pd
import tiktoken
import torch

from llm.classification import (
    SpamDataset,
    classify_text,
    configure_classifier_model,
    create_spam_dataloaders,
    prepare_spam_data,
)
from llm.instruction import (
    InstructionDataset,
    custom_collate_fn,
    format_prompt,
    generate_instruction_response,
    split_instruction_data,
)
from llm.model import GPTModel


TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def test_spam_data_preparation_and_loader(tmp_path):
    sms_path = tmp_path / "SMSSpamCollection.tsv"
    sms_path.write_text(
        "\n".join(
            [
                "ham\tI'll call you later",
                "spam\tWIN cash now",
                "ham\tAre we still meeting?",
                "spam\tClaim your prize today",
                "ham\tDinner at seven",
                "spam\tFree entry in contest",
            ]
        ),
        encoding="utf-8",
    )

    train_csv, val_csv, test_csv = prepare_spam_data(sms_path, tmp_path / "prepared")
    train_loader, val_loader, test_loader, max_length = create_spam_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=2,
    )

    input_batch, target_batch = next(iter(train_loader))
    assert input_batch.shape[0] == 2
    assert input_batch.shape[1] == max_length
    assert set(target_batch.tolist()).issubset({0, 1})
    assert len(val_loader.dataset) == 0
    assert len(test_loader.dataset) == 2


def test_classifier_head_checkpoint_round_trip(tmp_path):
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(TINY_CONFIG)
    configure_classifier_model(model, emb_dim=TINY_CONFIG["emb_dim"])
    weights_path = tmp_path / "spam-classifier.pth"
    torch.save(model.state_dict(), weights_path)
    (tmp_path / "spam-classifier.pth.json").write_text(
        json.dumps(
            {
                "task": "spam-classification",
                "preset": "tiny",
                "num_classes": 2,
                "max_length": 8,
                "labels": {"0": "ham", "1": "spam"},
            }
        ),
        encoding="utf-8",
    )

    loaded = GPTModel(TINY_CONFIG)
    configure_classifier_model(loaded, emb_dim=TINY_CONFIG["emb_dim"])
    loaded.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    predicted_id, probabilities = classify_text(
        model=loaded,
        text="WIN cash now",
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=8,
    )

    assert predicted_id in (0, 1)
    assert probabilities.shape == (2,)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_spam_dataset_truncates_and_pads_to_max_length(tmp_path):
    csv_path = tmp_path / "spam.csv"
    pd.DataFrame(
        [
            {"Label": 0, "Text": "short"},
            {"Label": 1, "Text": "this is a longer text message"},
        ]
    ).to_csv(csv_path, index=False)

    dataset = SpamDataset(csv_path, tokenizer=tiktoken.get_encoding("gpt2"), max_length=4)
    encoded, label = dataset[1]

    assert encoded.shape == (4,)
    assert label.item() == 1


def test_instruction_dataset_collate_masks_extra_padding():
    data = [
        {"instruction": "Say hello", "input": "", "output": "Hello."},
        {"instruction": "Summarize", "input": "A longer input text", "output": "Short."},
    ]
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionDataset(data, tokenizer)

    inputs, targets = custom_collate_fn([dataset[0], dataset[1]])

    assert inputs.shape == targets.shape
    assert inputs.shape[0] == 2
    assert (targets == -100).any()


def test_instruction_prompt_format_and_split():
    data = [
        {"instruction": str(index), "input": "", "output": str(index)}
        for index in range(10)
    ]

    train_data, test_data, val_data = split_instruction_data(data, train_portion=0.6, test_portion=0.2)

    assert len(train_data) == 6
    assert len(test_data) == 2
    assert len(val_data) == 2
    assert "### Instruction:\nRewrite this" in format_prompt("Rewrite this")


def test_instruction_response_generation_smoke():
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = GPTModel(TINY_CONFIG)
    model.eval()

    response = generate_instruction_response(
        model=model,
        instruction="Say hello",
        prompt_input="",
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        context_size=TINY_CONFIG["context_length"],
        max_new_tokens=2,
    )

    assert isinstance(response, str)
