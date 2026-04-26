from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from importlib import resources as package_resources
from pathlib import Path
from typing import Optional

import httpx
import typer

from .config import (
    DEFAULT_MODEL,
    config_dir,
    config_path,
    delete_file_config,
    get_config,
    load_file_config,
    redacted_key,
    save_file_config,
)


app = typer.Typer(no_args_is_help=True, help="Tarka command line tools.")
keys_app = typer.Typer(no_args_is_help=True, help="Manage Tarka API keys.")
training_app = typer.Typer(no_args_is_help=True, help="Operator-assisted training helpers.")
target_app = typer.Typer(no_args_is_help=True, help="Manage remote training targets.")
app.add_typer(keys_app, name="keys")
app.add_typer(training_app, name="training")
training_app.add_typer(target_app, name="target")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = "/data/tarka-training"
SPINNER_FRAMES = "|/-\\"


def emit(data: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    elif isinstance(data, str):
        typer.echo(data)
    else:
        for key, value in data.items():
            typer.echo(f"{key}: {value}")


def fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def update_config(updates: dict[str, object]) -> Path:
    data = load_file_config()
    data.update(updates)
    return save_file_config(data)


def openai_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def default_training_root() -> str:
    return os.environ.get("TARKA_TRAINING_ROOT", DEFAULT_TRAINING_ROOT)


def resource_path(kind: str, name: str) -> Path:
    repo_path = REPO_ROOT / kind / name
    if repo_path.exists():
        return repo_path

    resource = package_resources.files("tarka_cli").joinpath("resources", kind, name)
    if not resource.is_file():
        fail(f"Packaged resource not found: {kind}/{name}", 1)

    cache_dir = config_dir() / "resources" / kind
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    content = resource.read_bytes()
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
        if kind == "scripts":
            path.chmod(0o755)
    return path


def script_path(name: str) -> Path:
    return resource_path("scripts", name)


def get_training_targets() -> dict[str, dict[str, object]]:
    targets = load_file_config().get("training_targets", {})
    if not isinstance(targets, dict):
        return {}
    return {str(name): dict(value) for name, value in targets.items() if isinstance(value, dict)}


def get_default_training_target() -> str:
    value = load_file_config().get("default_training_target", "")
    return str(value) if value else ""


def save_training_targets(targets: dict[str, dict[str, object]], default: str = "") -> Path:
    data = load_file_config()
    data["training_targets"] = targets
    if default:
        data["default_training_target"] = default
    elif "default_training_target" in data and data["default_training_target"] not in targets:
        data.pop("default_training_target", None)
    return save_file_config(data)


def resolve_target(name: Optional[str]) -> tuple[str, dict[str, object]]:
    target_name = name or get_default_training_target()
    if not target_name:
        fail("Training target is required. Run `tarka training target add ...` first.", 2)
    targets = get_training_targets()
    if target_name not in targets:
        fail(f"Training target not found: {target_name}", 2)
    return target_name, targets[target_name]


def target_value(target: dict[str, object], key: str, default: str = "") -> str:
    value = target.get(key, default)
    return str(value) if value is not None else default


def ssh_destination(target: dict[str, object]) -> str:
    host = target_value(target, "host")
    user = target_value(target, "user")
    if not host or not user:
        fail("Target is missing host or user.", 2)
    return f"{user}@{host}"


def ssh_prefix(target: dict[str, object]) -> list[str]:
    args = ["ssh"]
    port = int(target.get("port", 22))
    if port != 22:
        args.extend(["-p", str(port)])
    args.append(ssh_destination(target))
    return args


def rsync_remote(target: dict[str, object], path: str) -> str:
    port = int(target.get("port", 22))
    dest = f"{ssh_destination(target)}:{path}"
    if port == 22:
        return dest
    return dest


def rsync_prefix(target: dict[str, object]) -> list[str]:
    args = ["rsync"]
    port = int(target.get("port", 22))
    if port != 22:
        args.extend(["-e", f"ssh -p {port}"])
    return args


def remote_workspace(target: dict[str, object]) -> str:
    workspace = target_value(target, "workspace")
    if workspace:
        return workspace.rstrip("/")
    root = target_value(target, "root", default_training_root()).rstrip("/")
    org = target_value(target, "org")
    if not org:
        fail("Target is missing workspace and org.", 2)
    return f"{root}/users/{org}"


def remote_root(target: dict[str, object]) -> str:
    root = target_value(target, "root")
    if root:
        return root.rstrip("/")
    workspace = remote_workspace(target)
    if "/users/" in workspace:
        return workspace.rsplit("/users/", 1)[0]
    fail("Target is missing root and workspace is not under <root>/users/<org>.", 2)


def remote_org(target: dict[str, object], override: Optional[str] = None) -> str:
    if override:
        return override
    org = target_value(target, "org")
    if org:
        return org
    workspace = remote_workspace(target)
    return workspace.rstrip("/").split("/")[-1]


def remote_path(target: dict[str, object], value: str | Path) -> str:
    raw = str(value)
    if raw.startswith("/"):
        return raw
    return f"{remote_workspace(target)}/{raw.lstrip('/')}"


def remote_tarka(target: dict[str, object]) -> str:
    command = target_value(target, "remote_tarka", "tarka")
    if command.startswith("~/"):
        user = target_value(target, "user")
        if user:
            return f"/home/{user}/{command[2:]}"
    return command


def run_remote(target: dict[str, object], command: list[str]) -> int:
    remote_command = shlex.join(command)
    result = subprocess.run(ssh_prefix(target) + [remote_command])
    return result.returncode


def run_remote_shell(target: dict[str, object], script: str) -> int:
    result = subprocess.run(ssh_prefix(target) + ["bash", "-s"], input=script, text=True)
    return result.returncode


def ensure_remote_dir(target: dict[str, object], path: str) -> None:
    result = run_remote(target, ["mkdir", "-p", path])
    if result != 0:
        raise typer.Exit(result)


def training_workspace(root: Path, org_slug: str) -> Path:
    return root / "users" / org_slug


def format_elapsed(started_at: float) -> str:
    seconds = int(time.monotonic() - started_at)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def clear_status_line() -> None:
    typer.echo("\r" + (" " * 120) + "\r", err=True, nl=False)


def workspace_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    hf_home = workspace / "scratch" / "huggingface"
    env.update(
        {
            "WORKSPACE": str(workspace),
            "HF_HOME": str(hf_home),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def abs_workspace_path(workspace: str | Path, value: str | Path) -> str:
    raw = str(value)
    if raw.startswith("/"):
        return raw
    return f"{str(workspace).rstrip('/')}/{raw.lstrip('/')}"


def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def hf_stage_shell_script(
    *,
    workspace: str,
    destination: str,
    dataset_id: str,
    split: str,
    val_split: str,
    text_column: str,
    output_format: str,
    limit: int,
    val_rows: int,
    max_chars: int,
    max_val_chars: int,
    shuffle_buffer: int,
    seed: int,
    revision: str,
    token_env: str,
    python_bin: str,
    trust_remote_code: bool,
    install_deps: bool,
) -> str:
    if output_format not in {"jsonl", "parquet", "nanochat-parquet"}:
        fail("--format must be one of: jsonl, parquet, nanochat-parquet", 2)
    if limit < 1:
        fail("--limit must be at least 1", 2)
    if val_rows < 0:
        fail("--val-rows cannot be negative", 2)

    config = {
        "dataset_id": dataset_id,
        "split": split,
        "val_split": val_split,
        "text_column": text_column,
        "output_format": output_format,
        "limit": limit,
        "val_rows": val_rows,
        "max_chars": max_chars,
        "max_val_chars": max_val_chars,
        "shuffle_buffer": shuffle_buffer,
        "seed": seed,
        "revision": revision,
        "token_env": token_env,
        "trust_remote_code": trust_remote_code,
    }
    maybe_install = (
        f"{shlex.quote(python_bin)} -m pip install --user -q datasets pyarrow\n"
        if install_deps
        else ""
    )
    return f"""set -euo pipefail
{shell_assign("WORKSPACE", workspace)}
{shell_assign("DEST", destination)}
{shell_assign("PYTHON_BIN", python_bin)}
if [[ "${{PYTHON_BIN}}" != /* && "${{PYTHON_BIN}}" == */* ]]; then
  PYTHON_BIN="${{WORKSPACE}}/${{PYTHON_BIN}}"
fi
export HF_HOME="${{WORKSPACE}}/scratch/huggingface"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets"
export HF_HUB_DISABLE_XET=1
mkdir -p "${{DEST}}" "${{HF_HOME}}" "${{HF_DATASETS_CACHE}}"
{maybe_install}"${{PYTHON_BIN}}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG = json.loads({json.dumps(json.dumps(config, sort_keys=True))})
DEST = Path(os.environ["DEST"])

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'datasets'. Install it in the active remote "
        "environment or rerun with --install-deps."
    ) from exc


def iter_texts(split: str, row_limit: int, char_limit: int, seed: int):
    kwargs = {{
        "split": split,
        "streaming": True,
    }}
    if CONFIG["revision"]:
        kwargs["revision"] = CONFIG["revision"]
    if CONFIG["trust_remote_code"]:
        kwargs["trust_remote_code"] = True
    token = os.environ.get(CONFIG["token_env"], "").strip()
    if token:
        kwargs["token"] = token

    dataset = load_dataset(CONFIG["dataset_id"], **kwargs)
    if CONFIG["shuffle_buffer"] > 0:
        dataset = dataset.shuffle(buffer_size=CONFIG["shuffle_buffer"], seed=seed)

    rows = 0
    chars = 0
    for example in dataset:
        value = example.get(CONFIG["text_column"])
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        yield text
        rows += 1
        chars += len(text)
        if rows >= row_limit or (char_limit > 0 and chars >= char_limit):
            break


def collect(split: str, row_limit: int, char_limit: int, seed: int) -> list[str]:
    rows = list(iter_texts(split, row_limit, char_limit, seed))
    if not rows:
        raise SystemExit(
            f"No text rows collected from {{CONFIG['dataset_id']}} split={{split}} "
            f"column={{CONFIG['text_column']}}."
        )
    return rows


def write_jsonl(rows: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in rows:
            handle.write(json.dumps({{"text": text}}, ensure_ascii=False) + "\\n")


def write_parquet(rows: list[str], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing Python package 'pyarrow'. Use --format jsonl, install pyarrow, "
            "or rerun with --install-deps."
        ) from exc
    table = pa.table({{"text": rows}})
    pq.write_table(table, path, row_group_size=128)


def write_manifest(files: list[Path], train_rows: int, val_rows: int = 0) -> None:
    manifest = {{
        "dataset": CONFIG["dataset_id"],
        "split": CONFIG["split"],
        "val_split": CONFIG["val_split"] if val_rows else None,
        "text_column": CONFIG["text_column"],
        "format": CONFIG["output_format"],
        "train_rows": train_rows,
        "val_rows": val_rows,
        "files": [str(path) for path in files],
        "hf_home": os.environ.get("HF_HOME"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
    }}
    (DEST / "tarka_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


fmt = CONFIG["output_format"]
DEST.mkdir(parents=True, exist_ok=True)

if fmt == "jsonl":
    rows = collect(CONFIG["split"], CONFIG["limit"], CONFIG["max_chars"], CONFIG["seed"])
    path = DEST / "data.jsonl"
    write_jsonl(rows, path)
    write_manifest([path], len(rows))
    print(f"Wrote {{path}} ({{len(rows)}} rows)")
elif fmt == "parquet":
    rows = collect(CONFIG["split"], CONFIG["limit"], CONFIG["max_chars"], CONFIG["seed"])
    path = DEST / "shard_00000.parquet"
    write_parquet(rows, path)
    write_manifest([path], len(rows))
    print(f"Wrote {{path}} ({{len(rows)}} rows)")
else:
    train_rows = collect(CONFIG["split"], CONFIG["limit"], CONFIG["max_chars"], CONFIG["seed"])
    val_rows = collect(
        CONFIG["val_split"],
        CONFIG["val_rows"],
        CONFIG["max_val_chars"],
        CONFIG["seed"] + 1,
    ) if CONFIG["val_rows"] else []
    train_path = DEST / "shard_00000.parquet"
    val_path = DEST / "shard_99999.parquet"
    write_parquet(train_rows, train_path)
    files = [train_path]
    if val_rows:
        write_parquet(val_rows, val_path)
        files.append(val_path)
    write_manifest(files, len(train_rows), len(val_rows))
    print(f"Wrote {{train_path}} ({{len(train_rows)}} train rows)")
    if val_rows:
        print(f"Wrote {{val_path}} ({{len(val_rows)}} val rows)")
PY
find "${{DEST}}" -maxdepth 1 -type f -print -exec ls -lh {{}} \\;
"""


def clone_repo_shell_script(*, repo_url: str, destination: str, ref: str, delete: bool) -> str:
    delete_line = 'git clean -fdx\n' if delete else ""
    checkout_line = f"git checkout --detach {shlex.quote(ref)}\n" if ref else ""
    return f"""set -euo pipefail
{shell_assign("REPO_URL", repo_url)}
{shell_assign("DEST", destination)}
mkdir -p "$(dirname "${{DEST}}")"
if [[ -d "${{DEST}}/.git" ]]; then
  cd "${{DEST}}"
  git fetch --all --tags --prune
else
  git clone "${{REPO_URL}}" "${{DEST}}"
  cd "${{DEST}}"
fi
{checkout_line}{delete_line}git rev-parse HEAD | tee .tarka_commit
git status --short
"""


def stream_training_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    follow: bool,
    spinner: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    output: queue.Queue[Optional[str]] = queue.Queue()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output.put(line)
        output.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    typer.echo(f"run pid: {process.pid}")
    typer.echo(f"log: {log_path}")

    frame_index = 0
    reader_done = False
    last_output_at = time.monotonic()
    last_status_len = 0
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n# command: {shlex.join(command)}\n")
            log_file.write(f"# cwd: {cwd}\n")
            log_file.write(f"# started_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
            while True:
                try:
                    item = output.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None and reader_done:
                        break
                    if spinner and time.monotonic() - last_output_at >= 0.6:
                        frame = SPINNER_FRAMES[frame_index % len(SPINNER_FRAMES)]
                        frame_index += 1
                        status = f"{frame} running {format_elapsed(started_at)} | pid {process.pid} | log {log_path}"
                        last_status_len = max(last_status_len, len(status))
                        typer.echo("\r" + status.ljust(last_status_len), err=True, nl=False)
                    continue

                if item is None:
                    reader_done = True
                    if process.poll() is not None:
                        break
                    continue

                last_output_at = time.monotonic()
                log_file.write(item)
                log_file.flush()
                if follow:
                    if spinner:
                        clear_status_line()
                        last_status_len = 0
                    typer.echo(item, nl=False)

            return_code = process.wait()
            log_file.write(f"\n# finished_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            log_file.write(f"# exit_code: {return_code}\n")
    except KeyboardInterrupt:
        clear_status_line()
        typer.echo("interrupt received; terminating training command", err=True)
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise typer.Exit(130)

    if spinner:
        clear_status_line()
    typer.echo(f"exit_code: {return_code}")
    return return_code


@app.command()
def init(
    key: Optional[str] = typer.Option(None, "--key", help="Tarka API key."),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="OpenAI-compatible API base URL."),
    app_url: Optional[str] = typer.Option(None, "--app-url", help="Tarka app URL."),
    model: Optional[str] = typer.Option(None, "--model", help="Default model name."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Create local CLI config."""
    current = get_config()
    path = update_config(
        {
            "api_key": key if key is not None else current["api_key"],
            "api_url": api_url if api_url is not None else current["api_url"],
            "app_url": app_url if app_url is not None else current["app_url"],
            "model": model if model is not None else current["model"],
            "http_timeout": current["http_timeout"],
        }
    )
    data = get_config()
    payload = {
        "config_path": str(path),
        "api_url": data["api_url"],
        "app_url": data["app_url"],
        "model": data["model"],
        "api_key": redacted_key(data["api_key"]),
        "next": 'tarka chat "hello"',
    }
    emit(payload, json_output)


@target_app.command("add")
def target_add(
    name: str,
    host: str = typer.Option(..., "--host"),
    user: str = typer.Option(..., "--user"),
    org: str = typer.Option(..., "--org", help="Training org slug."),
    workspace: Optional[str] = typer.Option(None, "--workspace"),
    root: Optional[str] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
    port: int = typer.Option(22, "--port"),
    remote_tarka_cmd: str = typer.Option("tarka", "--remote-tarka", help="Remote tarka command/path."),
    make_default: bool = typer.Option(True, "--default/--no-default"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Add or update a remote training target."""
    targets = get_training_targets()
    root_value = (root or default_training_root()).rstrip("/")
    workspace_value = workspace or f"{root_value}/users/{org}"
    if remote_tarka_cmd.startswith("~/"):
        remote_tarka_cmd = f"/home/{user}/{remote_tarka_cmd[2:]}"
    targets[name] = {
        "host": host,
        "user": user,
        "org": org,
        "root": root_value,
        "workspace": workspace_value.rstrip("/"),
        "port": port,
        "remote_tarka": remote_tarka_cmd,
    }
    default = name if make_default else get_default_training_target()
    save_training_targets(targets, default)
    payload = {"name": name, "default": default == name, "target": targets[name]}
    emit(payload, json_output)


@target_app.command("list")
def target_list(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """List remote training targets."""
    targets = get_training_targets()
    default = get_default_training_target()
    if json_output:
        emit({"default": default, "targets": targets}, True)
        return
    if not targets:
        typer.echo("No training targets configured.")
        return
    for name, target in sorted(targets.items()):
        marker = "*" if name == default else " "
        typer.echo(
            f"{marker} {name}: {target_value(target, 'user')}@{target_value(target, 'host')} "
            f"{target_value(target, 'workspace')}"
        )


@target_app.command("show")
def target_show(
    name: Optional[str] = typer.Argument(None),
    json_output: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Show a remote training target."""
    target_name, target = resolve_target(name)
    emit({"name": target_name, "target": target}, json_output)


@target_app.command("default")
def target_default(name: str) -> None:
    """Set the default remote training target."""
    targets = get_training_targets()
    if name not in targets:
        fail(f"Training target not found: {name}", 2)
    save_training_targets(targets, name)
    typer.echo(f"Default training target: {name}")


@target_app.command("remove")
def target_remove(name: str) -> None:
    """Remove a remote training target."""
    targets = get_training_targets()
    if name not in targets:
        fail(f"Training target not found: {name}", 2)
    targets.pop(name)
    default = get_default_training_target()
    save_training_targets(targets, "" if default == name else default)
    typer.echo(f"Removed training target: {name}")


@target_app.command("ssh")
def target_ssh(name: Optional[str] = typer.Argument(None)) -> None:
    """Open an interactive SSH session to a training target."""
    _, target = resolve_target(name)
    result = subprocess.run(ssh_prefix(target))
    raise typer.Exit(result.returncode)


@target_app.command("info")
def target_info(name: Optional[str] = typer.Argument(None)) -> None:
    """Print SSH and workspace details for a target."""
    target_name, target = resolve_target(name)
    typer.echo(f"name: {target_name}")
    typer.echo(f"ssh: ssh {ssh_destination(target)}")
    typer.echo(f"org: {remote_org(target)}")
    typer.echo(f"root: {remote_root(target)}")
    typer.echo(f"workspace: {remote_workspace(target)}")
    typer.echo(f"remote_tarka: {remote_tarka(target)}")


@app.command()
def login(
    key: Optional[str] = typer.Option(None, "--key", help="Tarka API key."),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="OpenAI-compatible API base URL."),
    app_url: Optional[str] = typer.Option(None, "--app-url", help="Tarka app URL."),
    model: Optional[str] = typer.Option(None, "--model", help="Default model name."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Alias for init."""
    init(key=key, api_url=api_url, app_url=app_url, model=model, json_output=json_output)


@app.command()
def logout(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Remove local CLI config."""
    removed = delete_file_config()
    emit({"removed": removed, "config_path": str(config_path())}, json_output)


@app.command()
def whoami(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Show local config identity context."""
    cfg = get_config()
    payload = {
        "api_base": cfg["api_url"],
        "app_url": cfg["app_url"],
        "model": cfg["model"],
        "api_key": redacted_key(cfg["api_key"]),
        "key_status": "configured" if cfg["api_key"] else "missing",
    }
    emit(payload, json_output)


@app.command()
def models(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """List models from the API, falling back to the configured default."""
    cfg = get_config()
    url = cfg["api_url"].rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=cfg["http_timeout"]) as client:
            response = client.get(url, headers=openai_headers(cfg["api_key"]))
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        data = {
            "object": "list",
            "fallback": True,
            "error": str(exc),
            "data": [{"id": cfg["model"] or DEFAULT_MODEL, "object": "model"}],
        }
    if json_output:
        emit(data, True)
        return
    for model_info in data.get("data", []):
        typer.echo(model_info.get("id", model_info))
    if data.get("fallback"):
        typer.echo(f"fallback: {data.get('error')}", err=True)


@app.command()
def status(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Unauthenticated health check against the configured API."""
    cfg = get_config()
    base = cfg["api_url"].rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    checks = {}
    with httpx.Client(timeout=cfg["http_timeout"]) as client:
        for name, url in {
            "gateway_models": base + "/models",
            "health": root + "/health",
        }.items():
            try:
                response = client.get(url, headers=openai_headers(cfg["api_key"]))
                checks[name] = {"ok": response.status_code < 500, "status_code": response.status_code}
            except Exception as exc:
                checks[name] = {"ok": False, "error": str(exc)}
    overall = any(check.get("ok") for check in checks.values())
    payload = {"ok": overall, "api_url": cfg["api_url"], "checks": checks}
    emit(payload, json_output)
    if not overall:
        raise typer.Exit(1)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User prompt."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name."),
    system: Optional[str] = typer.Option(None, "--system", help="System message."),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming."),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON response."),
) -> None:
    """Send one chat completion request."""
    cfg = get_config()
    if not cfg["api_key"]:
        fail("TARKA_API_KEY or local config api_key is required.", 2)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model or cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": not no_stream and not json_output,
    }
    if temperature is not None:
        body["temperature"] = temperature
    url = cfg["api_url"].rstrip("/") + "/chat/completions"
    try:
        with httpx.Client(timeout=None if body["stream"] else cfg["http_timeout"]) as client:
            if body["stream"]:
                with client.stream("POST", url, headers=openai_headers(cfg["api_key"]), json=body) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        data = json.loads(chunk)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            typer.echo(content, nl=False)
                    typer.echo()
            else:
                response = client.post(url, headers=openai_headers(cfg["api_key"]), json=body)
                response.raise_for_status()
                data = response.json()
                if json_output:
                    emit(data, True)
                else:
                    typer.echo(data["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as exc:
        fail(f"HTTP {exc.response.status_code}: {exc.response.text}", 1)
    except Exception as exc:
        fail(str(exc), 1)


@app.command()
def usage(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Placeholder for tarka-cloud usage API integration."""
    cfg = get_config()
    payload = {
        "implemented": False,
        "reason": "Usage lives in tarka-cloud; wire this after a stable API endpoint exists.",
        "app_url": cfg["app_url"],
        "suggested_url": cfg["app_url"].rstrip("/") + "/app/usage",
    }
    emit(payload, json_output)
    raise typer.Exit(2)


@keys_app.command("list")
def keys_list(json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Placeholder for tarka-cloud key API integration."""
    cfg = get_config()
    payload = {
        "implemented": False,
        "reason": "Key management lives in tarka-cloud; wire this after API endpoints are exposed.",
        "suggested_url": cfg["app_url"].rstrip("/") + "/app/keys",
    }
    emit(payload, json_output)
    raise typer.Exit(2)


@keys_app.command("create")
def keys_create(alias: str, json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Placeholder for tarka-cloud key creation."""
    cfg = get_config()
    emit({"implemented": False, "alias": alias, "suggested_url": cfg["app_url"].rstrip("/") + "/app/keys"}, json_output)
    raise typer.Exit(2)


@keys_app.command("revoke")
def keys_revoke(alias: str, json_output: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """Placeholder for tarka-cloud key revocation."""
    cfg = get_config()
    emit({"implemented": False, "alias": alias, "suggested_url": cfg["app_url"].rstrip("/") + "/app/keys"}, json_output)
    raise typer.Exit(2)


@training_app.command("init")
def training_init(
    output: Path = typer.Option(Path("training_request.md"), "--output", "-o", help="Output request file."),
    yaml_example: bool = typer.Option(False, "--yaml-example", help="Use YAML example instead of markdown template."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output."),
) -> None:
    """Create a local training request file."""
    source = resource_path("docs", "training_request.example.yaml" if yaml_example else "training_request_template.md")
    if output.exists() and not force:
        fail(f"Refusing to overwrite existing file: {output}. Use --force.", 2)
    shutil.copyfile(source, output)
    typer.echo(f"Created {output}")


@training_app.command("handoff")
def training_handoff(
    output: Path = typer.Option(Path("training_customer_handoff.md"), "--output", "-o"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output."),
) -> None:
    """Create a customer handoff note from the template."""
    source = resource_path("docs", "training_customer_handoff_template.md")
    if output.exists() and not force:
        fail(f"Refusing to overwrite existing file: {output}. Use --force.", 2)
    shutil.copyfile(source, output)
    typer.echo(f"Created {output}")


@training_app.command("check")
def training_check(
    org_slug: Optional[str] = typer.Argument(None),
    run_name: str = typer.Option("preflight", "--run-name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    expect_gpu: bool = typer.Option(True, "--expect-gpu/--skip-gpu"),
    expect_docker: bool = typer.Option(True, "--expect-docker/--skip-docker"),
    expect_data: bool = typer.Option(True, "--expect-data/--skip-data"),
    expect_repo: bool = typer.Option(False, "--expect-repo/--skip-repo"),
    min_free_gb: int = typer.Option(100, "--min-free-gb"),
) -> None:
    """Run workspace preflight checks."""
    training_root = root or Path(default_training_root())
    if target:
        _, remote = resolve_target(target)
        org = remote_org(remote, org_slug)
        command = [
            remote_tarka(remote),
            "training",
            "check",
            org,
            "--root",
            remote_root(remote),
            "--run-name",
            run_name,
            "--min-free-gb",
            str(min_free_gb),
        ]
        command.append("--expect-gpu" if expect_gpu else "--skip-gpu")
        command.append("--expect-docker" if expect_docker else "--skip-docker")
        command.append("--expect-data" if expect_data else "--skip-data")
        command.append("--expect-repo" if expect_repo else "--skip-repo")
        raise typer.Exit(run_remote(remote, command))

    if not org_slug:
        fail("ORG_SLUG is required unless --target is provided.", 2)
    env = os.environ.copy()
    env.update(
        {
            "TARKA_TRAINING_ROOT": str(training_root),
            "TARKA_EXPECT_GPU": "1" if expect_gpu else "0",
            "TARKA_EXPECT_DOCKER": "1" if expect_docker else "0",
            "TARKA_EXPECT_DATA": "1" if expect_data else "0",
            "TARKA_EXPECT_REPO": "1" if expect_repo else "0",
            "TARKA_MIN_FREE_GB": str(min_free_gb),
        }
    )
    result = subprocess.run([str(script_path("training_preflight.sh")), org_slug, run_name], env=env)
    raise typer.Exit(result.returncode)


@training_app.command("summary")
def training_summary(
    org_slug: str,
    run_name: str,
    root: Optional[Path] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
    workload: Optional[str] = typer.Option(None, "--workload"),
    tool: Optional[str] = typer.Option(None, "--tool"),
    repo: Optional[Path] = typer.Option(None, "--repo"),
    data_source: Optional[str] = typer.Option(None, "--data-source"),
    entrypoint: Optional[str] = typer.Option(None, "--entrypoint"),
    exit_status: Optional[str] = typer.Option(None, "--exit-status"),
) -> None:
    """Generate a markdown run summary."""
    training_root = root or Path(default_training_root())
    env = os.environ.copy()
    env["TARKA_TRAINING_ROOT"] = str(training_root)
    if workload:
        env["TARKA_WORKLOAD"] = workload
    if tool:
        env["TARKA_TOOL"] = tool
    if repo:
        env["TARKA_REPO"] = str(repo)
    if data_source:
        env["TARKA_DATA_SOURCE"] = data_source
    if entrypoint:
        env["TARKA_ENTRYPOINT"] = entrypoint
    if exit_status:
        env["TARKA_EXIT_STATUS"] = exit_status
    result = subprocess.run([str(script_path("create_training_run_summary.sh")), org_slug, run_name], env=env)
    raise typer.Exit(result.returncode)


@training_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def training_run(
    ctx: typer.Context,
    org_slug: Optional[str] = typer.Argument(None),
    run_name: str = typer.Option("run-001", "--run-name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Command working directory."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    detach: bool = typer.Option(False, "--detach", help="Start remote run in tmux and return."),
    no_follow: bool = typer.Option(False, "--no-follow", help="Write log but do not stream command output."),
    no_spinner: bool = typer.Option(False, "--no-spinner", help="Disable running status indicator."),
) -> None:
    """Run a training command inside a Phase 0 workspace."""
    training_root = root or Path(default_training_root())
    command = list(ctx.args)

    if target:
        _, remote = resolve_target(target)
        org = remote_org(remote)
        if org_slug:
            if org_slug == org and command:
                # Backward compatibility for the old
                # `training run <org> --target ... -- <command>` form.
                pass
            else:
                command.insert(0, org_slug)
        if not command:
            fail("Training command is required after '--'.", 2)
        remote_cwd = remote_path(remote, cwd or ".")
        remote_command = [
            remote_tarka(remote),
            "training",
            "run",
            org,
            "--root",
            remote_root(remote),
            "--run-name",
            run_name,
            "--cwd",
            remote_cwd,
        ]
        if no_follow:
            remote_command.append("--no-follow")
        if no_spinner:
            remote_command.append("--no-spinner")
        remote_command.append("--")
        remote_command.extend(command)
        if detach:
            tmux_command = ["tmux", "new-session", "-d", "-s", run_name, shlex.join(remote_command)]
            return_code = run_remote(remote, tmux_command)
            if return_code == 0:
                typer.echo(f"started remote session: {run_name}")
                typer.echo(f"logs: tarka training logs --target {target} --run-name {run_name} --follow")
            raise typer.Exit(return_code)
        raise typer.Exit(run_remote(remote, remote_command))

    if not command:
        fail("Training command is required after '--'.", 2)
    if not org_slug:
        fail("ORG_SLUG is required unless --target is provided.", 2)
    if detach:
        fail("--detach is only supported with --target.", 2)
    workspace = training_workspace(training_root, org_slug)
    if not workspace.exists():
        fail(f"Workspace does not exist: {workspace}", 1)
    command_cwd = cwd or workspace
    if not command_cwd.exists():
        fail(f"Working directory does not exist: {command_cwd}", 1)
    log_path = workspace / "logs" / f"{run_name}.log"
    env = workspace_env(workspace)
    env["TARKA_RUN_NAME"] = run_name
    return_code = stream_training_process(
        command=command,
        cwd=command_cwd,
        env=env,
        log_path=log_path,
        follow=not no_follow,
        spinner=not no_spinner,
    )
    raise typer.Exit(return_code)


@training_app.command("monitor")
def training_monitor(
    org_slug: Optional[str] = typer.Argument(None),
    run_name: str = typer.Option("run-001", "--run-name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    lines: int = typer.Option(40, "--lines", min=0),
    seconds: Optional[int] = typer.Option(None, "--seconds", min=1, help="Stop monitoring after this many seconds."),
    no_spinner: bool = typer.Option(False, "--no-spinner"),
) -> None:
    """Follow a training log with a lightweight running indicator."""
    training_root = root or Path(default_training_root())
    if target:
        _, remote = resolve_target(target)
        org = remote_org(remote, org_slug)
        command = [
            remote_tarka(remote),
            "training",
            "monitor",
            org,
            "--root",
            remote_root(remote),
            "--run-name",
            run_name,
            "--lines",
            str(lines),
        ]
        if seconds is not None:
            command.extend(["--seconds", str(seconds)])
        if no_spinner:
            command.append("--no-spinner")
        raise typer.Exit(run_remote(remote, command))

    if not org_slug:
        fail("ORG_SLUG is required unless --target is provided.", 2)
    workspace = training_workspace(training_root, org_slug)
    log_path = workspace / "logs" / f"{run_name}.log"
    if not log_path.exists():
        fail(f"Log file does not exist: {log_path}", 1)

    if lines > 0:
        previous = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        for line in previous:
            typer.echo(line)

    started_at = time.monotonic()
    frame_index = 0
    last_status_len = 0
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(0, os.SEEK_END)
            while True:
                if seconds is not None and time.monotonic() - started_at >= seconds:
                    if not no_spinner:
                        clear_status_line()
                    return
                line = log_file.readline()
                if line:
                    if not no_spinner:
                        clear_status_line()
                        last_status_len = 0
                    typer.echo(line, nl=False)
                    continue
                if not no_spinner:
                    frame = SPINNER_FRAMES[frame_index % len(SPINNER_FRAMES)]
                    frame_index += 1
                    checkpoints = workspace / "checkpoints"
                    status = f"{frame} monitoring {format_elapsed(started_at)} | log {log_path}"
                    if checkpoints.exists():
                        status += f" | checkpoints {checkpoints}"
                    last_status_len = max(last_status_len, len(status))
                    typer.echo("\r" + status.ljust(last_status_len), err=True, nl=False)
                time.sleep(0.5)
    except KeyboardInterrupt:
        if not no_spinner:
            clear_status_line()
        raise typer.Exit(130)


@training_app.command("upload")
def training_upload(
    source: Path,
    dest: str = typer.Option("datasets/main", "--to", help="Workspace-relative destination."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
) -> None:
    """Upload a file or directory to a remote training workspace."""
    if not target:
        fail("--target is required for upload.", 2)
    if not source.exists():
        fail(f"Source does not exist: {source}", 1)
    _, remote = resolve_target(target)
    destination = remote_path(remote, dest)
    ensure_remote_dir(remote, destination)
    source_arg = str(source)
    if source.is_dir() and not source_arg.endswith("/"):
        source_arg += "/"
    result = subprocess.run(
        rsync_prefix(remote) + ["-az", "--progress", source_arg, rsync_remote(remote, destination.rstrip("/") + "/")]
    )
    raise typer.Exit(result.returncode)


@training_app.command("stage-hf")
def training_stage_hf(
    dataset_id: str = typer.Argument(..., help="Hugging Face dataset id, for example org/name."),
    org_slug: Optional[str] = typer.Option(None, "--org", help="Local org slug. Inferred from --target when possible."),
    dest: str = typer.Option("datasets/main", "--to", help="Workspace-relative destination."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    root: Optional[Path] = typer.Option(None, "--root", help="Local training root."),
    split: str = typer.Option("train", "--split", help="Training split."),
    val_split: str = typer.Option("test", "--val-split", help="Validation split for nanochat-parquet."),
    text_column: str = typer.Option("text", "--text-column", help="Column containing training text."),
    output_format: str = typer.Option("jsonl", "--format", help="jsonl, parquet, or nanochat-parquet."),
    limit: int = typer.Option(1000, "--limit", min=1, help="Train row limit."),
    val_rows: int = typer.Option(100, "--val-rows", min=0, help="Validation row limit for nanochat-parquet."),
    max_chars: int = typer.Option(0, "--max-chars", min=0, help="Train character cap; 0 disables."),
    max_val_chars: int = typer.Option(0, "--max-val-chars", min=0, help="Validation character cap; 0 disables."),
    shuffle_buffer: int = typer.Option(10000, "--shuffle-buffer", min=0),
    seed: int = typer.Option(42, "--seed"),
    revision: Optional[str] = typer.Option(None, "--revision", help="Dataset revision/commit."),
    token_env: str = typer.Option("HF_TOKEN", "--token-env", help="Remote env var containing a Hugging Face token."),
    python_bin: str = typer.Option("python3", "--python", help="Remote/local Python executable."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    install_deps: bool = typer.Option(False, "--install-deps", help="Run `python -m pip install --user datasets pyarrow` first."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the staging script without running it."),
) -> None:
    """Stage a bounded Hugging Face dataset sample into the workspace."""
    if output_format not in {"jsonl", "parquet", "nanochat-parquet"}:
        fail("--format must be one of: jsonl, parquet, nanochat-parquet", 2)

    if target:
        target_name, remote = resolve_target(target)
        workspace = remote_workspace(remote)
        destination = remote_path(remote, dest)
        script = hf_stage_shell_script(
            workspace=workspace,
            destination=destination,
            dataset_id=dataset_id,
            split=split,
            val_split=val_split,
            text_column=text_column,
            output_format=output_format,
            limit=limit,
            val_rows=val_rows,
            max_chars=max_chars,
            max_val_chars=max_val_chars,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            revision=revision or "",
            token_env=token_env,
            python_bin=python_bin,
            trust_remote_code=trust_remote_code,
            install_deps=install_deps,
        )
        if dry_run:
            typer.echo(f"# target: {target_name}")
            typer.echo(f"# destination: {ssh_destination(remote)}:{destination}")
            typer.echo(script)
            return
        raise typer.Exit(run_remote_shell(remote, script))

    if not org_slug:
        fail("--org is required unless --target is provided.", 2)
    workspace_path = training_workspace(root or Path(default_training_root()), org_slug)
    destination = abs_workspace_path(workspace_path, dest)
    script = hf_stage_shell_script(
        workspace=str(workspace_path),
        destination=destination,
        dataset_id=dataset_id,
        split=split,
        val_split=val_split,
        text_column=text_column,
        output_format=output_format,
        limit=limit,
        val_rows=val_rows,
        max_chars=max_chars,
        max_val_chars=max_val_chars,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        revision=revision or "",
        token_env=token_env,
        python_bin=python_bin,
        trust_remote_code=trust_remote_code,
        install_deps=install_deps,
    )
    if dry_run:
        typer.echo(script)
        return
    result = subprocess.run(["bash", "-s"], input=script, text=True)
    raise typer.Exit(result.returncode)


@training_app.command("clone-repo")
def training_clone_repo(
    repo_url: str = typer.Argument(..., help="Git URL for the training code repository."),
    dest: Optional[str] = typer.Option(None, "--to", help="Workspace-relative destination. Defaults to repos/<repo-name>."),
    ref: Optional[str] = typer.Option(None, "--ref", help="Commit, tag, or branch to check out."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    org_slug: Optional[str] = typer.Option(None, "--org", help="Local org slug. Inferred from --target when possible."),
    root: Optional[Path] = typer.Option(None, "--root", help="Local training root."),
    delete: bool = typer.Option(False, "--delete", help="Clean untracked files after checkout."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the clone/update script without running it."),
) -> None:
    """Clone or update a training repository inside the workspace."""
    repo_name = Path(repo_url.rstrip("/").removesuffix(".git")).name or "training-repo"
    destination_relative = dest or f"repos/{repo_name}"

    if target:
        target_name, remote = resolve_target(target)
        destination = remote_path(remote, destination_relative)
        script = clone_repo_shell_script(
            repo_url=repo_url,
            destination=destination,
            ref=ref or "",
            delete=delete,
        )
        if dry_run:
            typer.echo(f"# target: {target_name}")
            typer.echo(f"# destination: {ssh_destination(remote)}:{destination}")
            typer.echo(script)
            return
        raise typer.Exit(run_remote_shell(remote, script))

    if not org_slug:
        fail("--org is required unless --target is provided.", 2)
    workspace_path = training_workspace(root or Path(default_training_root()), org_slug)
    destination = abs_workspace_path(workspace_path, destination_relative)
    script = clone_repo_shell_script(
        repo_url=repo_url,
        destination=destination,
        ref=ref or "",
        delete=delete,
    )
    if dry_run:
        typer.echo(script)
        return
    result = subprocess.run(["bash", "-s"], input=script, text=True)
    raise typer.Exit(result.returncode)


@training_app.command("sync-repo")
def training_sync_repo(
    source: Path = typer.Argument(Path(".")),
    dest: Optional[str] = typer.Option(None, "--to", help="Workspace-relative destination."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    delete: bool = typer.Option(False, "--delete", help="Delete remote files that no longer exist locally."),
) -> None:
    """Sync a local training repo to a remote workspace."""
    if not target:
        fail("--target is required for sync-repo.", 2)
    if not source.exists() or not source.is_dir():
        fail(f"Source directory does not exist: {source}", 1)
    _, remote = resolve_target(target)
    repo_name = source.resolve().name
    destination = remote_path(remote, dest or f"repos/{repo_name}")
    ensure_remote_dir(remote, destination)
    args = rsync_prefix(remote) + [
        "-az",
        "--progress",
        "--exclude",
        "__pycache__/",
        "--exclude",
        ".venv/",
        "--exclude",
        ".pytest_cache/",
    ]
    if delete:
        args.append("--delete")
    args.extend([str(source).rstrip("/") + "/", rsync_remote(remote, destination.rstrip("/") + "/")])
    result = subprocess.run(args)
    raise typer.Exit(result.returncode)


@training_app.command("logs")
def training_logs(
    org_slug: Optional[str] = typer.Argument(None),
    run_name: str = typer.Option("run-001", "--run-name"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    lines: int = typer.Option(80, "--lines", min=0),
    follow: bool = typer.Option(False, "--follow", "-f"),
) -> None:
    """Read or follow a remote training log."""
    if not target:
        fail("--target is required for logs. Use `training monitor` for local logs.", 2)
    _, remote = resolve_target(target)
    org = remote_org(remote, org_slug)
    log_path = f"{remote_root(remote).rstrip('/')}/users/{org}/logs/{run_name}.log"
    command = ["tail"]
    if follow:
        command.append("-f")
    command.extend(["-n", str(lines), log_path])
    raise typer.Exit(run_remote(remote, command))


@training_app.command("pull-artifacts")
def training_pull_artifacts(
    output: Path = typer.Option(Path("artifacts"), "--output", "-o"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Remote training target."),
    remote_dir: str = typer.Option("artifacts", "--from", help="Workspace-relative artifact directory."),
) -> None:
    """Download artifact bundles from a remote training workspace."""
    if not target:
        fail("--target is required for pull-artifacts.", 2)
    _, remote = resolve_target(target)
    output.mkdir(parents=True, exist_ok=True)
    source = remote_path(remote, remote_dir).rstrip("/") + "/"
    result = subprocess.run(rsync_prefix(remote) + ["-az", "--progress", rsync_remote(remote, source), str(output) + "/"])
    raise typer.Exit(result.returncode)


@training_app.command("artifacts")
def training_artifacts(
    path: Path = typer.Option(..., "--path", "-p", help="Artifact directory or workspace path."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """List artifact bundles and checksums."""
    artifact_dir = path / "artifacts" if (path / "artifacts").is_dir() else path
    if not artifact_dir.exists():
        fail(f"Artifact path does not exist: {artifact_dir}", 1)
    items = []
    for bundle in sorted(artifact_dir.glob("*.tar.gz")):
        checksum = bundle.with_suffix(bundle.suffix + ".sha256")
        items.append(
            {
                "bundle": str(bundle),
                "bytes": bundle.stat().st_size,
                "checksum": str(checksum) if checksum.exists() else "",
            }
        )
    if json_output:
        emit({"artifacts": items}, True)
    else:
        if not items:
            typer.echo(f"No artifact bundles found in {artifact_dir}")
        for item in items:
            typer.echo(f"{item['bundle']} ({item['bytes']} bytes)")
            if item["checksum"]:
                typer.echo(f"  checksum: {item['checksum']}")


@training_app.command("ssh-info")
def training_ssh_info(
    org_slug: str,
    host: str = typer.Option("<training-host>", "--host"),
    user: Optional[str] = typer.Option(None, "--user"),
    root: Optional[Path] = typer.Option(None, "--root", help="Training root. Defaults to TARKA_TRAINING_ROOT or /data/tarka-training."),
) -> None:
    """Print SSH/workspace instructions for an operator-assisted training user."""
    training_root = root or Path(default_training_root())
    ssh_user = user or f"tarka_{org_slug.replace('-', '_')}"
    workspace = training_root / "users" / org_slug
    typer.echo(f"ssh: ssh {ssh_user}@{host}")
    typer.echo(f"workspace: {workspace}")
    typer.echo(f"datasets: {workspace / 'datasets'}")
    typer.echo(f"repos: {workspace / 'repos'}")
    typer.echo(f"logs: {workspace / 'logs'}")
    typer.echo(f"artifacts: {workspace / 'artifacts'}")


if __name__ == "__main__":
    app()
