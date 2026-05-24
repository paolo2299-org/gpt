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

Generate with saved weights:

```bash
.venv/bin/python scripts/generate.py --weights model.pth --prompt "Every effort moves you"
.venv/bin/python scripts/generate.py --weights model.best.pth --prompt "Every effort moves you"
```

The default preset is `demo-small`, which keeps the chapter 5 context length and
training methodology while reducing model dimensions for practical demos. Use
`--preset book-124M` for the closer chapter 5 model shape.

## Tests

```bash
.venv/bin/python -m pytest tests
```
