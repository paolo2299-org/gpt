# LLM From Scratch Demo

This project adapts the chapter 5 pretraining workflow from Sebastian Raschka's
`LLMs-from-scratch` repository into a small, scriptable demo.

## Scripts

Create an untrained model with deterministic random weights:

```bash
.venv/bin/python scripts/random_demo.py
```

Pretrain on the book text, or pass your own text file:

```bash
.venv/bin/python scripts/pretrain.py --output model.pth
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
.venv/bin/python scripts/generate.py --weights model.pth --prompt "Every effort moves you"
.venv/bin/python scripts/generate.py --weights model.best.pth --prompt "Every effort moves you"
```

Run the local web UI directly from the virtualenv:

```bash
make flask-run
```

Run the web UI with Docker Compose:

```bash
make run
```

The web app loads `model.jane-austen-5.pth` on boot by default. Override that with
`MODEL_WEIGHTS_PATH=/path/to/model.pth` and, if needed, `MODEL_PRESET=book-124M`.

The default preset is `demo-small`, which keeps the chapter 5 context length and
training methodology while reducing model dimensions for practical demos. Use
`--preset book-124M` for the closer chapter 5 model shape.

## Tests

```bash
.venv/bin/python -m pytest tests
```

## Docker And Deployment

This app follows the same VPS deployment shape as the other `pdlawson.com`
projects:

- Docker Compose service name: `gpt`
- Public host: `gpt.pdlawson.com`
- Production compose files: `compose.yml` + `compose.prod.yml`
- External reverse-proxy network: `web`
- Expected VPS working directory: `/srv/gpt/app/large-language-model`
- Expected model path on VPS: `/srv/gpt/models/model.jane-austen-5.pth`

The model weights are not baked into the Docker image. Copy
`model.jane-austen-5.pth` to the VPS path above before starting the production
service.

Production commands on the VPS:

```bash
make prod-start
make prod-stop
make prod-restart
```

Pushing to `main` runs `.github/workflows/deploy.yml`, builds the image, pushes
it to GHCR, then SSHs into the VPS and restarts the `gpt` service.
