from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tarka_cli import main
from tarka_cli.main import app


runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Tarka command line tools" in result.output


def test_training_help_exits_zero() -> None:
    result = runner.invoke(app, ["training", "--help"])

    assert result.exit_code == 0
    assert "Operator-assisted training helpers" in result.output


def test_init_writes_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))

    result = runner.invoke(app, ["init", "--key", "tk_test_1234567890", "--json"])

    assert result.exit_code == 0
    assert "tk_test" in result.output
    assert (tmp_path / "config" / "config.json").exists()


def test_training_target_add_uses_env_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("TARKA_TRAINING_ROOT", "/tmp/tarka-training")

    result = runner.invoke(
        app,
        [
            "training",
            "target",
            "add",
            "dgx",
            "--host",
            "gpu.example.com",
            "--user",
            "alice",
            "--org",
            "customer-one",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "/tmp/tarka-training/users/customer-one" in result.output


def test_training_init_works_without_repo_docs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(main, "REPO_ROOT", tmp_path / "missing-repo")
    output = tmp_path / "training_request.md"

    result = runner.invoke(app, ["training", "init", "--output", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    assert "workload_type" in output.read_text()


def test_training_stage_hf_dry_run_uses_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))

    add = runner.invoke(
        app,
        [
            "training",
            "target",
            "add",
            "dgx",
            "--host",
            "gpu.example.com",
            "--user",
            "alice",
            "--org",
            "customer-one",
        ],
    )
    assert add.exit_code == 0

    result = runner.invoke(
        app,
        [
            "training",
            "stage-hf",
            "IRIIS-RESEARCH/Nepali-Text-Corpus",
            "--target",
            "dgx",
            "--to",
            "scratch/nanochat/base_data_climbmix",
            "--text-column",
            "Article",
            "--format",
            "nanochat-parquet",
            "--limit",
            "100",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "gpu.example.com" in result.output
    assert "IRIIS-RESEARCH/Nepali-Text-Corpus" in result.output
    assert "scratch/nanochat/base_data_climbmix" in result.output
    assert "nanochat-parquet" in result.output


def test_training_clone_repo_dry_run_defaults_repo_destination(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))

    add = runner.invoke(
        app,
        [
            "training",
            "target",
            "add",
            "dgx",
            "--host",
            "gpu.example.com",
            "--user",
            "alice",
            "--org",
            "customer-one",
        ],
    )
    assert add.exit_code == 0

    result = runner.invoke(
        app,
        [
            "training",
            "clone-repo",
            "https://github.com/karpathy/nanochat.git",
            "--target",
            "dgx",
            "--ref",
            "main",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "git clone" in result.output
    assert "repos/nanochat" in result.output
    assert "git checkout --detach main" in result.output


def test_keys_list_points_to_app(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARKA_CONFIG_DIR", str(tmp_path / "config"))

    result = runner.invoke(app, ["keys", "list"])

    assert result.exit_code == 2
    assert "/app/keys" in result.output
