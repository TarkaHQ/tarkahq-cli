#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: install.sh [source]

Installs the Tarka CLI into a user-local virtualenv and links the
`tarka` executable into ~/.local/bin.

Source defaults to the current git checkout when run from this repo.
For published installs, pass a Git repository URL.

Examples:
  ./install.sh
  ./install.sh https://github.com/TarkaHQ/tarkahq-cli.git
  curl -fsSL https://install.tarkahq.com/cli | bash

Environment:
  TARKA_CLI_SOURCE       install source if no argument is provided
  TARKA_CLI_REF          git ref to checkout, default: main
  TARKA_CLI_HOME         default: ~/.local/share/tarka-cli
  TARKA_CLI_BIN_DIR      default: ~/.local/bin
  PYTHON                 default: python3
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON:-python3}"
CLI_HOME="${TARKA_CLI_HOME:-${HOME}/.local/share/tarka-cli}"
BIN_DIR="${TARKA_CLI_BIN_DIR:-${HOME}/.local/bin}"
VENV_DIR="${CLI_HOME}/venv"
SOURCE_DIR="${CLI_HOME}/source"
GIT_REF="${TARKA_CLI_REF:-main}"
SOURCE="${1:-${TARKA_CLI_SOURCE:-}}"

if [[ -z "${SOURCE}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/pyproject.toml" && -d "${SCRIPT_DIR}/tarka_cli" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
  if [[ -f "${REPO_ROOT}/pyproject.toml" && -d "${REPO_ROOT}/tarka_cli" ]]; then
    SOURCE="${REPO_ROOT}"
  else
    SOURCE="https://github.com/TarkaHQ/tarkahq-cli.git"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

resolve_source() {
  local source="$1"
  if [[ -d "${source}" && -f "${source}/pyproject.toml" ]]; then
    cd "${source}"
    pwd
    return
  fi

  if [[ "${source}" == git+* ]]; then
    source="${source#git+}"
  fi

  if [[ "${source}" == http://* || "${source}" == https://* || "${source}" == git@* || "${source}" == ssh://* ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "git is required to install from ${source}" >&2
      exit 2
    fi
    if [[ -d "${SOURCE_DIR}/.git" ]]; then
      git -C "${SOURCE_DIR}" fetch --tags origin
    else
      rm -rf "${SOURCE_DIR}"
      git clone "${source}" "${SOURCE_DIR}"
    fi
    git -C "${SOURCE_DIR}" checkout "${GIT_REF}" >/dev/null
    git -C "${SOURCE_DIR}" pull --ff-only origin "${GIT_REF}" >/dev/null 2>&1 || true
    echo "${SOURCE_DIR}"
    return
  fi

  echo "Unsupported install source: ${source}" >&2
  exit 2
}

mkdir -p "${CLI_HOME}" "${BIN_DIR}"
INSTALL_SOURCE="$(resolve_source "${SOURCE}")"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install --upgrade -e "${INSTALL_SOURCE}"

ln -sf "${VENV_DIR}/bin/tarka" "${BIN_DIR}/tarka"

cat <<EOF
Tarka CLI installed.

Binary:
  ${BIN_DIR}/tarka

Source:
  ${INSTALL_SOURCE}

Version check:
  tarka --help

If your shell cannot find tarka, add this to your shell profile:
  export PATH="${BIN_DIR}:\$PATH"
EOF
