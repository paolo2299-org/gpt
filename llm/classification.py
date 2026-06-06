from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from llm.model import GPTModel


SPAM_DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAM_DATA_BACKUP_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
LABEL_TO_ID = {"ham": 0, "spam": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def download_and_unzip_spam_data(
    url: str,
    zip_path: Path,
    extracted_path: Path,
    data_file_path: Path,
) -> None:
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    import requests

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with zip_path.open("wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def ensure_spam_tsv(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    extracted_path = data_dir / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"
    zip_path = data_dir / "sms_spam_collection.zip"

    try:
        download_and_unzip_spam_data(SPAM_DATA_URL, zip_path, extracted_path, data_file_path)
    except Exception as exc:
        print(f"Primary URL failed: {exc}. Trying backup URL...")
        download_and_unzip_spam_data(SPAM_DATA_BACKUP_URL, zip_path, extracted_path, data_file_path)

    return data_file_path


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(
    df: pd.DataFrame,
    train_frac: float,
    validation_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def prepare_spam_data(
    sms_tsv_path: Path,
    output_dir: Path,
    train_frac: float = 0.7,
    validation_frac: float = 0.1,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sms_tsv_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map(LABEL_TO_ID)

    train_df, validation_df, test_df = random_split(balanced_df, train_frac, validation_frac)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=None)
    validation_df.to_csv(val_path, index=None)
    test_df.to_csv(test_path, index=None)
    return train_path, val_path, test_path


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        tokenizer,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[: self.max_length] for encoded_text in self.encoded_texts]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_spam_dataloaders(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path | None = None,
    batch_size: int = 8,
    seed: int = 123,
) -> tuple[DataLoader, DataLoader, DataLoader | None, int]:
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(train_csv, tokenizer=tokenizer)
    val_dataset = SpamDataset(val_csv, max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = None
    if test_csv is not None:
        test_dataset = SpamDataset(test_csv, max_length=train_dataset.max_length, tokenizer=tokenizer)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader, train_dataset.max_length


def configure_classifier_model(
    model: GPTModel,
    emb_dim: int,
    num_classes: int = 2,
    train_last_block: bool = True,
) -> GPTModel:
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(in_features=emb_dim, out_features=num_classes)

    if train_last_block:
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    return model


def classify_text(
    model: GPTModel,
    text: str,
    tokenizer,
    device: torch.device,
    max_length: int,
    pad_token_id: int = 50256,
) -> tuple[int, torch.Tensor]:
    model.eval()
    encoded = tokenizer.encode(text)[:max_length]
    encoded = encoded + [pad_token_id] * (max_length - len(encoded))
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label, torch.softmax(logits, dim=-1).squeeze(0)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def calc_classifier_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_classifier_loss_loader(data_loader, model, device, num_batches=None):
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
        loss = calc_classifier_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_classifier(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_classifier_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_classifier_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    on_eval=None,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_classifier_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classifier(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
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
                        examples_seen=examples_seen,
                    )

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
