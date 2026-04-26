# Training Customer Handoff Template

Send this after Tarka has accepted the request and assigned a training
window.

## Training Window

```text
org:
node:
scheduled_start:
scheduled_end:
operator_contact:
```

This is an operator-assisted training window. During the window, Tarka
may pause inference on the assigned single-GPU node.

## Workspace

```text
ssh_user:
ssh_host:
workspace:
datasets_path:
repos_path:
checkpoints_path:
outputs_path:
logs_path:
scratch_path:
```

All job data, caches, logs, checkpoints, and outputs must stay under the
workspace.

## Data Handoff

Approved data staging method:

```text
method: Hugging Face | rsync/scp | rclone | other
source:
destination:
expected_size:
```

Examples:

```bash
rsync -av ./dataset/ <ssh_user>@<ssh_host>:<workspace>/datasets/main/
```

```bash
rclone copy remote:bucket/path <workspace>/datasets/main
```

For Hugging Face datasets or models that require authentication, provide
a short-lived token through the agreed secret channel. Do not put tokens
in Git, Slack, email, or markdown files.

## Code Handoff

```text
repo_url:
commit_or_tag:
config_path:
entrypoint_command:
```

If the repo is private, confirm access before the training window.

## Run Visibility

Tarka will provide:

```text
final command
log path
checkpoint path
artifact bundle path
artifact checksum
run summary
```

Optional live log path:

```bash
tail -f <logs_path>/<run-name>.log
```

CLI launch from the training node:

```bash
tarka training run <org> \
  --run-name <run-name> \
  --cwd <workspace>/repos/<repo> \
  -- <entrypoint_command>
```

CLI monitoring from the training node:

```bash
tarka training monitor <org> --run-name <run-name>
```

For long runs, launch inside `tmux` so the process survives SSH
disconnects.

## Completion

At the end of the run, Tarka will provide:

```text
exit status
artifact bundle
checksum
checkpoint location
known issues
recommended next run change
retention deadline
```
