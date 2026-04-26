from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://api.tarkahq.com/v1"
DEFAULT_APP_URL = "https://app.tarkahq.com"
DEFAULT_MODEL = "qwen3-coder-30b"


def config_dir() -> Path:
    override = os.environ.get("TARKA_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "tarka"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_file_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON at {path}: {exc}") from exc


def save_file_config(data: dict[str, Any]) -> Path:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return path


def delete_file_config() -> bool:
    path = config_path()
    if path.exists():
        path.unlink()
        return True
    return False


def get_config() -> dict[str, Any]:
    file_config = load_file_config()
    return {
        "api_key": os.environ.get("TARKA_API_KEY", file_config.get("api_key", "")),
        "api_url": os.environ.get("TARKA_API_URL", file_config.get("api_url", DEFAULT_API_URL)),
        "app_url": os.environ.get("TARKA_APP_URL", file_config.get("app_url", DEFAULT_APP_URL)),
        "model": os.environ.get("TARKA_MODEL", file_config.get("model", DEFAULT_MODEL)),
        "http_timeout": float(os.environ.get("TARKA_HTTP_TIMEOUT", file_config.get("http_timeout", 30))),
    }


def redacted_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return key[:4] + "..."
    return key[:8] + "..." + key[-4:]
