# Tarka Training Access Handoff

This document gives you access to an operator-assisted GPU training
window on a Tarka DGX Spark node.

This is not a managed training platform yet. You will get an isolated
workspace, local data staging, preflight checks, live logs, checkpoints,
and artifact handoff.

## Access Credentials

Fill these values before sending:

```text
org_slug: <org-slug>
ssh_user: tarka_<org_slug_with_underscores>
ssh_host: <dgx-host-or-ip>
training_root: /data/tarka-training
workspace: /data/tarka-training/users/<org-slug>
scheduled_start: <scheduled-start>
scheduled_end: <scheduled-end>
operator_contact: <operator-contact>
```

SSH command:

```bash
ssh tarka_<org_slug_with_underscores>@<dgx-host-or-ip>
```

Authentication is by SSH public key only. No password will be issued.

Add this DGX as a local training target:

```bash
tarka training target add dgx \
  --host <dgx-host-or-ip> \
  --user tarka_<org_slug_with_underscores> \
  --org <org-slug> \
  --root <training-root> \
  --workspace <workspace>
```

## Public Key Setup

Send Tarka your SSH public key before the training window:

```bash
cat ~/.ssh/id_ed25519.pub
```

If you do not have one:

```bash
ssh-keygen -t ed25519 -C "<your-email>"
cat ~/.ssh/id_ed25519.pub
```

Do not send your private key. Only send the `.pub` file contents.

## Workspace Layout

All job data, caches, logs, checkpoints, and outputs must stay under the
workspace.

```text
<workspace>/
  datasets/      staged training data
  repos/         training code
  checkpoints/   checkpoints
  outputs/       model outputs and eval outputs
  logs/          run logs
  scratch/       caches and temporary files
  artifacts/     final bundles
```

## Install The CLI

Install the pilot CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
tarka --help
```

This pilot installer currently uses the Python prototype. The long-term
install path will be a native `tarka` binary via Homebrew or a
release-backed curl installer.

For the first private pilot, Tarka may preinstall the CLI on the
training node. In that case, start with:

```bash
tarka --help
```

## Bring Data

Training data must be staged onto the GPU node before the run starts.
Do not stream training data over the WAN during training.

From your laptop:

```bash
tarka training upload ./dataset --target dgx --to datasets/main
```

From the DGX, if using object storage:

```bash
rclone copy remote:bucket/path <workspace>/datasets/main
```

For private Hugging Face datasets or models, provide a short-lived token
through the agreed secret channel. Do not put tokens in Git, Slack,
email, or markdown files.

## Bring Code

Sync your training repo into the workspace:

```bash
tarka training sync-repo . --target dgx --to repos/<repo-name>
```

Your training code should have one clear entrypoint command.

Examples:

```bash
axolotl train configs/train.yml
```

```bash
python train.py --config configs/train.yml
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

## Preflight

Run this before training:

```bash
tarka training check --target dgx
```

Example:

```bash
tarka training check acme --root /data/tarka-training
```

The preflight checks workspace directories, write access, disk, GPU
visibility, Docker GPU access, and data presence.

## Start Training

Start the training command:

```bash
tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/<repo-name> \
  -- <your-training-command>
```

Example:

```bash
tarka training run --target dgx \
  --run-name run-001 \
  --detach \
  --cwd repos/acme-train \
  -- axolotl train configs/train.yml
```

The CLI will stream output, show a lightweight running indicator, and
write the full log to:

```text
<workspace>/logs/run-001.log
```

`--detach` starts the remote command in `tmux` so the job can survive
SSH disconnects. Omit `--detach` for an interactive foreground run.

## Monitor Progress

Follow the run log:

```bash
tarka training monitor --target dgx --run-name run-001
```

Run a bounded status check:

```bash
tarka training monitor --target dgx --run-name run-001 --seconds 30
```

Direct log tail:

```bash
tarka training logs --target dgx --run-name run-001 --follow
```

GPU state:

```bash
tarka training target ssh dgx
nvidia-smi
```

Checkpoint/output size:

```bash
du -sh <workspace>/checkpoints <workspace>/outputs
```

## Completion

At the end of the run, Tarka will provide:

```text
exit status
final log path
checkpoint path
artifact bundle
checksum
run summary
retention deadline
```

List artifact bundles:

```bash
tarka training pull-artifacts --target dgx --output ./artifacts
```
