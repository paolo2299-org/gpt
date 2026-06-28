# LLM From Scratch Demo

This project adapts the models and code from Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository into a small, scriptable demo.

## Scripts

Create an untrained model with deterministic random weights:

```bash
.venv/bin/python scripts/random_demo.py
```

Pretrain on a text file:

```bash
.venv/bin/python scripts/pretrain.py --input-file path/to/text.txt --output model.pth
```

Pretraining saves the final weights to `model.pth` and the lowest-validation-loss
checkpoint to `model.best.pth`. Use `--best-output` to choose a different path.

For a larger plain-text corpus, around a few MB, this is a useful starting run:

```bash
.venv/bin/python scripts/pretrain.py \
  --preset 124M \
  --input-file path/to/larger-text.txt \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --eval-freq 100 \
  --eval-iter 20 \
  --output model.pth
```

Generate with saved weights:

```bash
.venv/bin/python scripts/generate.py --weights model.pth --prompt "I wandered lonely as a"
.venv/bin/python scripts/generate.py --weights model.best.pth --prompt "I wandered lonely as a"
```

## Load GPT-2 open weights (book ch5)

The same `GPTModel` that the scripts above train from scratch is also the network
OpenAI's GPT-2 weights load into — only the config preset differs (`gpt2-small`:
1024-token context, `qkv_bias=True`, no dropout). OpenAI's weights are published
as a PyTorch state dict matching this model, so downloading them needs nothing but
torch + requests (no TensorFlow). Fetch them once:

```bash
.venv/bin/python scripts/fetch_gpt2.py --size 124M
```

That writes `weights/gpt2-small.pth` (+ a `.json` sidecar recording the preset),
which the generator and web UI then load like any other checkpoint:

```bash
.venv/bin/python scripts/generate.py --weights weights/gpt2-small.pth \
  --prompt "Every effort moves you" --temperature 1.0 --top-k 50 --max-new-tokens 25
```

## Fine-tuning examples (book ch6 and ch7)

The chapter 6 and 7 examples start from a GPT-2 checkpoint, so fetch one first
if `weights/gpt2-small.pth` is not already present:

```bash
.venv/bin/python scripts/fetch_gpt2.py --size 124M
```

Fine-tune the chapter 6 SMS spam classifier:

```bash
.venv/bin/python scripts/finetune_spam_classifier.py \
  --base-weights weights/gpt2-small.pth \
  --output weights/spam-classifier.pth
```

That script downloads and prepares the SMS Spam Collection under `data/` unless
you pass `--sms-tsv path/to/SMSSpamCollection.tsv`. It saves final weights plus a
lowest-validation-loss checkpoint (`weights/spam-classifier.best.pth` by
default). Classify a message with the completed classifier:

```bash
.venv/bin/python scripts/classify_spam.py \
  --weights weights/spam-classifier.best.pth \
  --prompt "WINNER!! Claim your free prize now"
```

Fine-tune the chapter 7 instruction-following model:

```bash
.venv/bin/python scripts/finetune_instruction.py \
  --base-weights weights/gpt2-small.pth \
  --input-file ~/projects/LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data.json \
  --output weights/instruction-following.pth
```

Generate an instruction-following response from the completed model:

```bash
.venv/bin/python scripts/instruct.py \
  --weights weights/instruction-following.best.pth \
  --prompt "Convert the active sentence to passive voice." \
  --input "The chef cooked the meal."
```

Fine-tune GPT-2 small as a plain text completion model on Jane Austen:

```bash
.venv/bin/python scripts/clean_austen_text.py
.venv/bin/python scripts/finetune_llm.py \
  --base-weights weights/gpt2-small.pth \
  --input-file texts/jane-austen.cleaned.txt \
  --output weights/jane-austen-gpt2-small.pth \
  --num-epochs 1 \
  --learning-rate 5e-5 \
  --batch-size 1
```

That performs continued next-token training: it starts from GPT-2's open weights
instead of random initialization, then trains further on the cleaned six-novel
Austen corpus. It also saves the lowest-validation-loss checkpoint as
`weights/jane-austen-gpt2-small.best.pth`.

Generate with the fine-tuned completion model:

```bash
.venv/bin/python scripts/generate.py \
  --weights weights/jane-austen-gpt2-small.best.pth \
  --prompt "She had long suspected" \
  --temperature 0.8 \
  --top-k 50 \
  --max-new-tokens 80
```

Run the local web UI directly from the virtualenv:

```bash
make flask-run
```

Run the web UI with Docker Compose:

```bash
cp .env.example .env  # first time only; edit values as needed
make run
```

The Docker Compose setup reads environment variables from `.env`, so this file must exist before running `make run`, `make dev`, or `make docker-test`.

The web app hosts **multiple demos** from a single weights directory. Set
`WEIGHTS_DIR` to the folder of `.pth` files (default `weights`). Each demo
defined in `app/demos.py` names the weights file it needs (e.g.
`jane-austen-gpt2-small.best.pth` for JaneGPT, `gpt2-small.pth` for the GPT-2
demo) and appears on the landing page only when that file is present. Models
load lazily on a demo's first request.

The default training preset is `124M`; `demo-small` is a tiny shape for fast
checks, and the `gpt2-*` presets match OpenAI's open weights.

## Tests

```bash
.venv/bin/python -m pytest tests
```

## Docker and Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose setup, VPS configuration, and CI/CD details.
