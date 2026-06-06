from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from llm.generation import generate, text_to_token_ids, token_ids_to_text
from llm.training import calc_loss_loader, train_model_simple


DEFAULT_INSTRUCTION_DATA_PATH = (
    Path.home()
    / "projects"
    / "LLMs-from-scratch"
    / "ch07"
    / "01_main-chapter-code"
    / "instruction-data.json"
)


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def load_instruction_data(path: str | Path = DEFAULT_INSTRUCTION_DATA_PATH) -> list[dict]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        return json.load(file)


def split_instruction_data(
    data: list[dict],
    train_portion: float = 0.85,
    test_portion: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_end = int(len(data) * train_portion)
    test_end = train_end + int(len(data) * test_portion)
    return data[:train_end], data[train_end:test_end], data[test_end:]


def format_input(entry: dict) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def format_prompt(instruction: str, prompt_input: str = "") -> str:
    return format_input({"instruction": instruction, "input": prompt_input})


def create_instruction_dataloaders(
    train_data: list[dict],
    val_data: list[dict],
    batch_size: int,
    device: torch.device,
    allowed_max_length: int,
    seed: int = 123,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = tiktoken.get_encoding("gpt2")
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader


def finetune_instruction_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    seed: int = 123,
    on_eval=None,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    torch.manual_seed(seed)
    return train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context=start_context,
        tokenizer=tokenizer,
        on_eval=on_eval,
    )


def instruction_losses(train_loader, val_loader, model, device, eval_iter: int) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_instruction_response(
    model,
    instruction: str,
    prompt_input: str,
    tokenizer,
    device: torch.device,
    context_size: int,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = 50256,
) -> str:
    input_text = format_prompt(instruction, prompt_input)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text) :].replace("### Response:", "").strip()
    return response_text
