# Docker and Deployment

This app follows the same VPS deployment shape as the other `pdlawson.com`
projects:

- Docker Compose service name: `gpt`
- Public host: `gpt.pdlawson.com`
- External reverse-proxy network: `web`
- Expected weights directory on VPS: `/srv/gpt/weights`
- Root-owned runtime definition: `/opt/hetzner-vm-setup/apps/gpt`
- Root-only runtime environment: `/etc/hetzner-apps/gpt/container.env`

The model weights are not baked into the Docker image. Production mounts the
whole `/srv/gpt/weights` directory into the container at `/weights`. Set
`WEIGHTS_DIR` to that directory (default `/weights`). The app serves **multiple
demos**: each demo (see `app/demos.py`) names a weights file inside `WEIGHTS_DIR`
(e.g. `model.dickens.pth` for JaneGPT, `gpt2-small.pth` for the GPT-2 demo), and
a demo appears on the landing page only if its file is present. Models are loaded
lazily on a demo's first request, so the app starts even if some (or all) weight
files are missing.

Generate the GPT-2 open-weights checkpoint once with `scripts/fetch_gpt2.py`
(pure torch + requests — it downloads a pre-converted PyTorch state dict; run it
locally, not in the deployed image) and copy the resulting `gpt2-small.pth` into
the weights directory.

> **Migration note (breaking env change):** the previous single-model variables
> `MODEL_WEIGHTS_PATH`, `MODEL_PRESET`, `SITE_TITLE`, and `AUTHOR_NAME` are
> replaced by `WEIGHTS_DIR`. On the VPS, rename `/srv/gpt/models` to
> `/srv/gpt/weights` and update `.env` before deploying this revision.

The Compose files in this repository remain useful for local or manual
operation. The authoritative production definition lives in the host operations
repository so the root deployment service never executes files from an editable
application checkout.

Production commands on the VPS:

```bash
sudo systemctl start app-deploy@gpt.service
sudo systemctl status --no-pager app-deploy@gpt.service
sudo journalctl -u app-deploy@gpt.service -n 100 --no-pager
```

Pushing to `main` runs `.github/workflows/deploy.yml`, tests the application,
then publishes the image to GHCR with both the commit SHA and `main` tags. The
workflow stops there and has no SSH access to the VPS. A host timer independently
checks the public package, resolves `main` to an immutable digest, deploys it,
waits for a health check, and retains the previous digest for rollback.
