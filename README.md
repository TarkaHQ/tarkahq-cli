# Tarka CLI

Command line tools for Tarka inference checks and operator-assisted
training access.

## Install

Pilot installer:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | bash
```

Pinned pilot install:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | TARKA_CLI_REF=v0.1.2 bash
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
tarka training clone-repo https://github.com/karpathy/nanochat.git \
  --target dgx \
  --to repos/nanochat \
  --ref <pinned-commit>

tarka training run --target dgx \
  --run-name nanochat-setup \
  --cwd repos/nanochat \
  -- uv sync --extra gpu

tarka training stage-hf IRIIS-RESEARCH/Nepali-Text-Corpus \
  --target dgx \
  --to scratch/nanochat/base_data_climbmix \
  --text-column Article \
  --format nanochat-parquet \
  --python repos/nanochat/.venv/bin/python \
  --limit 1000 \
  --val-rows 100
```

Run preflight and launch:

```bash
tarka training check --target dgx

tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/nanochat \
  -- bash -lc 'source .venv/bin/activate && export NANOCHAT_BASE_DIR="$WORKSPACE/scratch/nanochat" TORCH_COMPILE_DISABLE=1 && OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --device-type=cuda --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 --eval-every=-1 --eval-tokens=512 --num-iterations=50 --run=dummy --model-tag=spark-smoke-d4-nepali --core-metric-every=-1 --sample-every=-1 --save-every=-1'
```

Monitor and retrieve outputs:

```bash
tarka training logs --target dgx --run-name run-001 --follow
tarka training pull-artifacts --target dgx --output ./artifacts
```

See:

- [`docs/training_customer_quickstart.md`](docs/training_customer_quickstart.md)
- [`docs/cli_commands.md`](docs/cli_commands.md)
