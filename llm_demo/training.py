from __future__ import annotations

from pathlib import Path

import tiktoken
import torch

from llm_demo.generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from llm_demo.model import GPTModel, create_dataloader_v1


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )
    model.train()
    return token_ids_to_text(token_ids, tokenizer)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    sample_tokens=50,
    on_eval=None,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                if on_eval is not None:
                    on_eval(
                        model=model,
                        epoch=epoch + 1,
                        global_step=global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        tokens_seen=tokens_seen,
                    )

        sample = generate_sample(model, tokenizer, device, start_context, max_new_tokens=sample_tokens)
        print(sample.replace("\n", " "))

    return train_losses, val_losses, track_tokens_seen


def read_text(path: str | Path) -> str:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        return file.read()


def create_train_val_loaders(text_data, gpt_config, settings):
    train_ratio = settings["train_ratio"]
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def pretrain_from_text(text_data, gpt_config, settings, device, seed=123, on_eval=None):
    torch.manual_seed(seed)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    train_loader, val_loader = create_train_val_loaders(text_data, gpt_config, settings)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=settings["eval_freq"],
        eval_iter=settings["eval_iter"],
        start_context=settings["start_context"],
        tokenizer=tokenizer,
        on_eval=on_eval,
    )
    return train_losses, val_losses, tokens_seen, model
