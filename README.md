# Tarka CLI

Command line tools for Tarka inference checks and operator-assisted
training access.

## Install

Pilot installer:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | bash
```

The short install domain `https://install.tarkahq.com/cli` is not live
yet; use the GitHub installer above for pilots.

If your shell cannot find `tarka`, add `~/.local/bin` to your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Inference

```bash
tarka init --key tk_live_...
tarka status
tarka models
tarka chat "hello"
```

## Training Access

Add your assigned GPU target:

```bash
tarka training target add dgx \
  --host <ssh_host> \
  --user <ssh_user> \
  --org <org-slug> \
  --root <training-root> \
  --workspace <workspace>
```

Stage data and code:

```bash
tarka training upload ./dataset --target dgx --to datasets/main
tarka training sync-repo . --target dgx --to repos/<repo>
```

Run preflight and launch:

```bash
tarka training check --target dgx

tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/<repo> \
  -- python train.py --config configs/train.yml
```

Monitor and retrieve outputs:

```bash
tarka training logs --target dgx --run-name run-001 --follow
tarka training pull-artifacts --target dgx --output ./artifacts
```

See:

- [`docs/training_customer_quickstart.md`](docs/training_customer_quickstart.md)
- [`docs/cli_commands.md`](docs/cli_commands.md)
