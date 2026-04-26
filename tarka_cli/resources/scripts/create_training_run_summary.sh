#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: create_training_run_summary.sh <org-slug> <run-name>

Creates a markdown run summary under:
  <workspace>/logs/<run-name>-summary.md

Environment:
  TARKA_TRAINING_ROOT    default: /data/tarka-training
  TARKA_WORKLOAD         optional
  TARKA_TOOL             optional
  TARKA_NODE             optional, default hostname
  TARKA_REPO             optional
  TARKA_COMMIT           optional
  TARKA_CONFIG           optional
  TARKA_ENTRYPOINT       optional
  TARKA_DATA_SOURCE      optional
  TARKA_EXIT_STATUS      optional
  TARKA_RETENTION_UNTIL  optional
USAGE
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

ORG_SLUG="$1"
RUN_NAME="$2"
TRAINING_ROOT="${TARKA_TRAINING_ROOT:-/data/tarka-training}"
WORKSPACE="${TRAINING_ROOT}/users/${ORG_SLUG}"
LOG_DIR="${WORKSPACE}/logs"
SUMMARY="${LOG_DIR}/${RUN_NAME}-summary.md"

if [[ ! "${ORG_SLUG}" =~ ^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$ ]]; then
  echo "Invalid org slug: ${ORG_SLUG}" >&2
  exit 2
fi

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid run name: ${RUN_NAME}" >&2
  exit 2
fi

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "Workspace does not exist: ${WORKSPACE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

artifact_bundle="$(find "${WORKSPACE}/artifacts" -maxdepth 1 -type f -name "${RUN_NAME}*.tar.gz" 2>/dev/null | sort | tail -1 || true)"
artifact_checksum=""
if [[ -n "${artifact_bundle}" && -f "${artifact_bundle}.sha256" ]]; then
  artifact_checksum="${artifact_bundle}.sha256"
fi

repo_commit="${TARKA_COMMIT:-}"
if [[ -z "${repo_commit}" && -n "${TARKA_REPO:-}" && -d "${TARKA_REPO}/.git" ]]; then
  repo_commit="$(git -C "${TARKA_REPO}" rev-parse HEAD 2>/dev/null || true)"
fi

cat > "${SUMMARY}" <<EOF
# Training Run Summary: ${RUN_NAME}

## Run

\`\`\`text
org: ${ORG_SLUG}
run_name: ${RUN_NAME}
workload: ${TARKA_WORKLOAD:-}
tool: ${TARKA_TOOL:-}
node: ${TARKA_NODE:-$(hostname)}
started_at:
finished_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
exit_status: ${TARKA_EXIT_STATUS:-}
operator:
\`\`\`

## Inputs

\`\`\`text
repo: ${TARKA_REPO:-}
commit: ${repo_commit}
config: ${TARKA_CONFIG:-}
entrypoint: ${TARKA_ENTRYPOINT:-}
data_source: ${TARKA_DATA_SOURCE:-}
staged_data_path: ${WORKSPACE}/datasets
secrets_used:
\`\`\`

## Environment

\`\`\`text
workspace: ${WORKSPACE}
python_or_image:
gpu:
driver:
cuda:
important_env:
  HF_HOME=${WORKSPACE}/scratch/huggingface
  HF_DATASETS_CACHE=${WORKSPACE}/scratch/huggingface/datasets
  HF_HUB_DISABLE_XET=1
\`\`\`

## Outputs

\`\`\`text
logs: ${WORKSPACE}/logs
checkpoints: ${WORKSPACE}/checkpoints
outputs: ${WORKSPACE}/outputs
artifact_bundle: ${artifact_bundle}
artifact_checksum: ${artifact_checksum}
retention_until: ${TARKA_RETENTION_UNTIL:-}
\`\`\`

## Result

\`\`\`text
what_worked:
what_failed:
operator_intervention:
customer_visible_notes:
next_run_change:
\`\`\`

## Commands

\`\`\`bash
# data staging

# dependency setup

# training

# artifact collection
\`\`\`
EOF

echo "Run summary: ${SUMMARY}"
