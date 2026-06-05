# Docker and Deployment

This app follows the same VPS deployment shape as the other `pdlawson.com`
projects:

- Docker Compose service name: `gpt`
- Public host: `gpt.pdlawson.com`
- Production compose files: `compose.yml` + `compose.prod.yml`
- External reverse-proxy network: `web`
- Expected VPS working directory: `/srv/gpt/app/gpt`
- Expected weights directory on VPS: `/srv/gpt/weights`

The model weights are not baked into the Docker image. Production mounts the
whole `/srv/gpt/weights` directory into the container at `/weights`. Set
`WEIGHTS_DIR` to that directory (default `/weights`). The app serves **multiple
demos**: each demo (see `app/demos.py`) names a weights file inside `WEIGHTS_DIR`
(e.g. `model.dickens.pth` for JaneGPT), and a demo appears on the landing page
only if its file is present. Models are loaded lazily on a demo's first request,
so the app starts even if some (or all) weight files are missing.

> **Migration note (breaking env change):** the previous single-model variables
> `MODEL_WEIGHTS_PATH`, `MODEL_PRESET`, `SITE_TITLE`, and `AUTHOR_NAME` are
> replaced by `WEIGHTS_DIR`. On the VPS, rename `/srv/gpt/models` to
> `/srv/gpt/weights` and update `.env` before deploying this revision.

Production commands on the VPS:

```bash
make prod-start
make prod-stop
make prod-restart
```

Pushing to `main` runs `.github/workflows/deploy.yml`, builds the image, pushes
it to GHCR, then SSHs into the VPS and restarts the `gpt` service.
