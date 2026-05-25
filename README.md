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
  --preset book-124M \
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

Run the local web UI directly from the virtualenv:

```bash
make flask-run
```

Run the web UI with Docker Compose:

```bash
make run
```

The web app requires `MODEL_WEIGHTS_PATH` to be set. Optionally set `MODEL_PRESET=book-124M` if the weights use the larger model shape.

The default preset is `book-124M`. Use `--preset demo-small` for a smaller model shape suited to quick demos.

## Tests

```bash
.venv/bin/python -m pytest tests
```

## Docker and Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose setup, VPS configuration, and CI/CD details.
