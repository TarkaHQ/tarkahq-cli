#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: training_preflight.sh <org-slug> [run-name]

Checks a Phase 0 training workspace before launching a job.

Environment:
  TARKA_TRAINING_ROOT    default: /data/tarka-training
  TARKA_MIN_FREE_GB      default: 100
  TARKA_EXPECT_GPU       default: 1
  TARKA_EXPECT_DOCKER    default: 1
  TARKA_EXPECT_DATA      default: 1
  TARKA_EXPECT_REPO      default: 0

Exit codes:
  0  pass
  1  one or more required checks failed
  2  usage/config error
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

ORG_SLUG="$1"
RUN_NAME="${2:-preflight}"
TRAINING_ROOT="${TARKA_TRAINING_ROOT:-/data/tarka-training}"
WORKSPACE="${TRAINING_ROOT}/users/${ORG_SLUG}"
MIN_FREE_GB="${TARKA_MIN_FREE_GB:-100}"
EXPECT_GPU="${TARKA_EXPECT_GPU:-1}"
EXPECT_DOCKER="${TARKA_EXPECT_DOCKER:-1}"
EXPECT_DATA="${TARKA_EXPECT_DATA:-1}"
EXPECT_REPO="${TARKA_EXPECT_REPO:-0}"
LOG_DIR="${WORKSPACE}/logs"
REPORT="${LOG_DIR}/${RUN_NAME}-preflight.txt"
FAILED=0

if [[ ! "${ORG_SLUG}" =~ ^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$ ]]; then
  echo "Invalid org slug: ${ORG_SLUG}" >&2
  exit 2
fi

status() {
  local state="$1"
  shift
  printf '[%s] %s\n' "${state}" "$*"
}

pass() {
  status "PASS" "$@"
}

warn() {
  status "WARN" "$@"
}

fail() {
  status "FAIL" "$@"
  FAILED=1
}

run_checks() {
  echo "Training preflight"
  echo "org_slug=${ORG_SLUG}"
  echo "workspace=${WORKSPACE}"
  echo "run_name=${RUN_NAME}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  if [[ -d "${WORKSPACE}" ]]; then
    pass "workspace exists"
  else
    fail "workspace missing: ${WORKSPACE}"
    return
  fi

  mkdir -p "${LOG_DIR}"

  for dir in datasets repos checkpoints outputs logs scratch artifacts; do
    if [[ -d "${WORKSPACE}/${dir}" ]]; then
      pass "directory exists: ${dir}"
    else
      fail "directory missing: ${dir}"
    fi
  done

  for dir in checkpoints outputs logs scratch artifacts; do
    if [[ -w "${WORKSPACE}/${dir}" ]]; then
      pass "directory writable: ${dir}"
    else
      fail "directory not writable: ${dir}"
    fi
  done

  if command -v df >/dev/null 2>&1; then
    local free_kb
    local free_gb
    free_kb="$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')"
    free_gb="$((free_kb / 1024 / 1024))"
    if [[ "${free_gb}" -ge "${MIN_FREE_GB}" ]]; then
      pass "free disk ${free_gb}GB >= ${MIN_FREE_GB}GB"
    else
      fail "free disk ${free_gb}GB < ${MIN_FREE_GB}GB"
    fi
  else
    warn "df not found; disk check skipped"
  fi

  if [[ "${EXPECT_GPU}" == "1" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      pass "nvidia-smi found"
      nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv,noheader || warn "nvidia-smi query failed"
    else
      fail "nvidia-smi not found"
    fi
  else
    warn "GPU check skipped"
  fi

  if [[ "${EXPECT_DOCKER}" == "1" ]]; then
    if command -v docker >/dev/null 2>&1; then
      pass "docker found"
      if docker run --rm --gpus all ubuntu nvidia-smi >/dev/null 2>&1; then
        pass "docker GPU smoke succeeded"
      else
        fail "docker GPU smoke failed"
      fi
    else
      fail "docker not found"
    fi
  else
    warn "docker GPU smoke skipped"
  fi

  if [[ "${EXPECT_DATA}" == "1" ]]; then
    if find "${WORKSPACE}/datasets" -mindepth 1 -maxdepth 2 -print -quit | grep -q .; then
      pass "dataset path has content"
      du -sh "${WORKSPACE}/datasets"/* 2>/dev/null || true
    else
      fail "dataset path is empty"
    fi
  else
    warn "dataset content check skipped"
  fi

  if [[ "${EXPECT_REPO}" == "1" ]]; then
    if find "${WORKSPACE}/repos" -mindepth 1 -maxdepth 2 -type d -name .git -print -quit | grep -q .; then
      pass "repo checkout found"
    else
      fail "repo checkout not found under repos/"
    fi
  else
    warn "repo checkout check skipped"
  fi

  echo
  echo "Recommended workspace env:"
  echo "export WORKSPACE=\"${WORKSPACE}\""
  echo "export HF_HOME=\"${WORKSPACE}/scratch/huggingface\""
  echo "export HF_DATASETS_CACHE=\"${WORKSPACE}/scratch/huggingface/datasets\""
  echo "export HF_HUB_DISABLE_XET=1"
}

mkdir -p "${LOG_DIR}" 2>/dev/null || true
run_checks > "${REPORT}"
cat "${REPORT}"

echo
echo "Preflight report: ${REPORT}"

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi
