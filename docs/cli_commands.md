# Tarka CLI Command Inventory

This is the working command surface for the first `tarka` CLI.

The first implementation is intentionally small:

- OpenAI-compatible inference calls go directly to the configured API
  base URL.
- Usage and key-management commands are placeholders until tarka-cloud
  exposes stable API endpoints.
- Training commands are Phase 0 operator helpers. They do not submit
  managed jobs or require scheduler state.

## Brand

```bash
tarka banner
```

Implemented in the scaffold:

- `tarka banner` prints the terminal version of the Tarka mountain mark
  and wordmark.

## Config

```bash
tarka init
tarka login
tarka logout
tarka whoami
```

Implemented in the scaffold:

- `tarka init --key <key> --api-url <url> --model <model>` stores local
  config.
- `tarka login` is an alias for `tarka init`.
- `tarka logout` removes local config.
- `tarka whoami --json` shows redacted key, API URL, app URL, and model.

Config defaults:

```text
config file: ~/.config/tarka/config.json
api url: https://api.tarkahq.com/v1
app url: https://app.tarkahq.com
model: qwen3-coder-30b
```

Environment overrides:

```text
TARKA_API_KEY
TARKA_API_URL
TARKA_APP_URL
TARKA_MODEL
TARKA_CONFIG_DIR
TARKA_HTTP_TIMEOUT
```

## Inference

```bash
tarka status
tarka models
tarka chat "hello"
```

Implemented in the scaffold:

- `tarka status --json` probes `/models` and `/health`.
- `tarka models --json` lists models from the API, with a fallback model
  when the API is unavailable.
- `tarka chat "hello"` calls `/chat/completions`.
- `tarka chat "hello" --no-stream --json` returns the raw response.

Common options:

```bash
tarka chat "PROMPT" --model qwen3-coder-30b
tarka chat "PROMPT" --system "You are concise."
tarka chat "PROMPT" --max-tokens 256
tarka chat "PROMPT" --temperature 0.2
```

## Usage

```bash
tarka usage
```

Current state:

- Placeholder command.
- Exits with code `2`.
- Points the user to the tarka-cloud usage page.

Planned fields once the tarka-cloud API is available:

- request count
- input tokens/chars
- output tokens
- latency
- model
- request type
- error count
- org/user attribution

## API Keys

```bash
tarka keys list
tarka keys create <alias>
tarka keys revoke <alias>
```

Current state:

- Placeholder commands.
- Exit with code `2`.
- Point the user to the tarka-cloud keys page.

Planned options:

```bash
tarka keys create laptop
tarka keys create ci --monthly-token-limit 1000000
tarka keys create evals --monthly-request-limit 5000
```

## Training: Phase 0 Helpers

```bash
tarka training init
tarka training handoff
tarka training target add <name>
tarka training upload ./dataset --target <name> --to datasets/main
tarka training stage-hf <hf-dataset-id> --target <name> --to datasets/main
tarka training clone-repo <git-url> --target <name> --to repos/<repo>
tarka training sync-repo . --target <name> --to repos/<repo>
tarka training check <org-slug>
tarka training check --target <name>
tarka training run <org-slug> -- <entrypoint command>
tarka training run --target <name> --detach -- <entrypoint command>
tarka training monitor <org-slug>
tarka training monitor --target <name>
tarka training logs --target <name> --follow
tarka training pull-artifacts --target <name>
tarka training summary <org-slug> <run-name>
tarka training artifacts --path <workspace-or-artifact-dir>
tarka training ssh-info <org-slug>
```

Implemented in the scaffold:

- `tarka training init` copies the training request template.
- `tarka training init --yaml-example` copies the example YAML request.
- `tarka training handoff` copies the customer handoff template.
- `tarka training target add dgx --host <host> --user <user> --org <org>`
  stores a remote DGX target.
- `tarka training upload ./dataset --target dgx --to datasets/main`
  stages data through `rsync`.
- `tarka training stage-hf IRIIS-RESEARCH/Nepali-Text-Corpus --target dgx --to scratch/nanochat/base_data_climbmix --text-column Article --format nanochat-parquet`
  streams a bounded Hugging Face dataset sample on the training node,
  writes workspace-local cache under `scratch/huggingface`, and creates
  either JSONL, parquet, or nanochat-compatible parquet shards.
- `tarka training clone-repo https://github.com/karpathy/nanochat.git --target dgx --to repos/nanochat --ref <commit>`
  clones or updates a remote training repo and records the checked-out
  commit in `.tarka_commit`.
- `tarka training sync-repo . --target dgx --to repos/<repo>` syncs a
  local training repo through `rsync`.
- `tarka training check <org> --root <path>` wraps
  `scripts/training_preflight.sh`.
- `tarka training check <org> --skip-gpu --skip-docker --skip-data`
  supports local/operator dry runs.
- `tarka training check --target dgx` runs the remote preflight helper
  over SSH.
- `tarka training run <org> --run-name <run> --cwd <repo> -- <command>`
  runs a foreground training command from the assigned training node,
  tees output to `<workspace>/logs/<run>.log`, and shows a spinner while
  the process is active.
- `tarka training run --target dgx --detach --cwd repos/<repo> -- <command>`
  starts that command remotely in `tmux`.
- `tarka training monitor <org> --run-name <run>` follows an existing
  training log with a lightweight running indicator.
- `tarka training monitor <org> --run-name <run> --seconds 30` runs a
  bounded status check.
- `tarka training logs --target dgx --run-name <run> --follow` tails a
  remote log over SSH.
- `tarka training pull-artifacts --target dgx --output ./artifacts`
  downloads remote artifacts through `rsync`.
- `tarka training summary <org> <run>` wraps
  `scripts/create_training_run_summary.sh`.
- `tarka training artifacts --path <workspace>` lists `.tar.gz` bundles
  and checksum files.
- `tarka training ssh-info <org> --host <host>` prints SSH and workspace
  paths.

These commands support the current manual training path: create an
intake file, provision or point at a workspace, run preflight, run the
training job by hand, bundle artifacts, and create a summary.

## Training: Managed Later

Do not build these until the Phase 0 path has repeat users or paid
commitment.

```bash
tarka training nodes
tarka training submit nanochat --config runs/tarka-smoke.sh
tarka training submit axolotl --config train.yml
tarka training submit unsloth --script train.py
tarka training jobs
tarka training logs <job-id>
tarka training artifacts <job-id>
tarka training cancel <job-id>
```

These require a training-access API, durable job state, and a node agent
or scheduler integration. They should not call the OpenAI-compatible
inference API.

## Exit Codes

```text
0 success
1 remote/runtime/network failure
2 auth/usage/config error or intentionally unimplemented placeholder
130 interrupted
```
