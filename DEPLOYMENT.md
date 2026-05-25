# Docker and Deployment

This app follows the same VPS deployment shape as the other `pdlawson.com`
projects:

- Docker Compose service name: `gpt`
- Public host: `gpt.pdlawson.com`
- Production compose files: `compose.yml` + `compose.prod.yml`
- External reverse-proxy network: `web`
- Expected VPS working directory: `/srv/gpt/app/gpt`
- Expected model directory on VPS: `/srv/gpt/models`

The model weights are not baked into the Docker image. Production mounts the
whole `/srv/gpt/models` directory into the container at `/models`. Set
`MODEL_WEIGHTS_PATH` to the path of the model file to load, for example
`MODEL_WEIGHTS_PATH=/models/model.pth`. This variable is required — the app
will not start without it.

Production commands on the VPS:

```bash
make prod-start
make prod-stop
make prod-restart
```

Pushing to `main` runs `.github/workflows/deploy.yml`, builds the image, pushes
it to GHCR, then SSHs into the VPS and restarts the `gpt` service.
