# Training Access Customer Quickstart

This is the customer-facing Phase 0 training flow.

Tarka Training Access is an operator-assisted GPU window. You bring a
repo, config, data source, and entrypoint command. Tarka provides an
isolated workspace, local data staging, preflight checks, logs,
checkpoints, and artifact handoff.

This is not a self-serve Slurm cluster or managed training scheduler
yet.

## 1. Send Tarka The Intake

Install the CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | bash
```

If your shell cannot find `tarka`, add `~/.local/bin` to your `PATH`.
This pilot installer currently uses the Python prototype. The intended
long-term install path is a native `tarka` binary via Homebrew or a
release-backed curl installer.

For the first private pilot, Tarka may preinstall the CLI on the
training node instead of asking you to install it from your laptop.

Create the request file:

```bash
tarka training init --output training_request.md
```

Fill in:

```text
workload_type: LoRA/QLoRA fine-tune | full fine-tune | pre-training | other
preferred_tool: Axolotl | Unsloth | nanochat | custom
base_model:
repo_url:
commit_or_tag:
config_path:
entrypoint_command:
custom_docker_image:
data_source:
dataset_size:
sample_size_for_smoke:
required_secrets:
expected_runtime:
expected_checkpoint_size:
artifact_handoff:
```

Do not put secrets in the request file. Tarka will agree on a separate
secret handoff path for Hugging Face, W&B, or object-store credentials.

## 2. Connect To The Training Node

Tarka will provide:

```text
ssh_user:
ssh_host:
workspace:
scheduled_start:
scheduled_end:
```

Add the DGX as a training target on your laptop:

```bash
tarka training target add dgx \
  --host <ssh_host> \
  --user <ssh_user> \
  --org <org-slug> \
  --root <training-root> \
  --workspace <workspace>
```

Confirm the target:

```bash
tarka training target info dgx
```

You can still open an SSH session if needed:

```bash
tarka training target ssh dgx
```

All remote data, code, logs, checkpoints, and caches must stay under:

```text
<workspace>
```

The standard workspace layout is:

```text
<workspace>/
  datasets/
  repos/
  checkpoints/
  outputs/
  logs/
  scratch/
  artifacts/
```

## 3. Confirm Data Is Local

Training data should be staged onto the GPU node before the run starts.
Do not stream training data over the WAN during training.

Examples:

```bash
tarka training upload ./dataset --target dgx --to datasets/main
```

```bash
rclone copy remote:bucket/path <workspace>/datasets/main
```

For Hugging Face datasets, Tarka will either stage the data during the
operator setup window or give you a shell with workspace-local cache
defaults.

## 4. Confirm Code And Command

Your training code should be in a repo with a pinned commit:

```text
repo_url:
commit_or_tag:
config_path:
entrypoint_command:
```

Sync the repo to the DGX workspace:

```bash
tarka training sync-repo . --target dgx --to repos/<repo>
```

Preferred tools:

```text
Axolotl: YAML-driven LoRA/QLoRA/full fine-tuning
Unsloth: optimized Python/notebook-derived fine-tuning
nanochat: small pre-training smoke tests
custom: accepted when the entrypoint and output paths are explicit
```

Custom training code must:

```text
take paths from args or env
write logs under <workspace>/logs
write checkpoints under <workspace>/checkpoints or a declared path
support a small smoke mode
support resume for long runs
avoid sudo and system-level mutation
keep caches under <workspace>/scratch
```

## 5. Run Preflight

From your laptop:

```bash
tarka training check --target dgx
```

This checks workspace directories, write access, disk, GPU visibility,
Docker GPU access, and data presence.

## 6. Start Training

For Phase 0, the CLI launches a foreground remote run over SSH:

```bash
tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/<repo> \
  -- <entrypoint command>
```

Example Axolotl run:

```bash
tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/acme-train \
  -- axolotl train configs/train.yml
```

Example Python run:

```bash
tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/acme-train \
  -- python train.py --config configs/train.yml
```

The command automatically sets:

```text
WORKSPACE=<workspace>
HF_HOME=<workspace>/scratch/huggingface
HF_DATASETS_CACHE=<workspace>/scratch/huggingface/datasets
HF_HUB_DISABLE_XET=1
TARKA_RUN_NAME=<run-name>
```

The CLI streams output and writes the full log to:

```text
<workspace>/logs/<run-name>.log
```

`--detach` starts the remote command in `tmux` so the job can survive
SSH disconnects. Omit `--detach` for an interactive foreground run.

## 7. Monitor Progress

The simplest monitor is:

```bash
tarka training monitor --target dgx --run-name run-001
```

For a bounded status check:

```bash
tarka training monitor --target dgx --run-name run-001 --seconds 30
```

You can also inspect logs directly:

```bash
tarka training logs --target dgx --run-name run-001 --follow
```

Or check GPU state:

```bash
tarka training target ssh dgx
nvidia-smi
```

For tools that support W&B, TensorBoard, or metrics files, configure
those from your training repo. Tarka's required baseline is the log file
plus checkpoints/artifacts under the workspace.

## 8. Receive Artifacts

After completion, Tarka will provide:

```text
exit status
final log path
checkpoint path
artifact bundle
checksum
run summary
retention deadline
```

You can list bundles with:

```bash
tarka training pull-artifacts --target dgx --output ./artifacts
```
