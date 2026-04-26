# CLI Distribution

The current CLI is a Python prototype. It is useful for validating the
command surface, but the long-term customer CLI should be a native
single-binary tool. Keep Python as a bootstrap path, not the final
product packaging strategy.

The developer install is:

```bash
pip install -e .
```

That is not suitable for a customer handoff. For pilots, use a
published `curl` installer or a GitHub release. Homebrew is the
preferred customer install path once the CLI is stable enough to
version.

## PyPI Namespace

The `tarka` package name is already occupied on PyPI, so it cannot be
reserved directly unless the owner transfers it. Use `tarkahq` for the
Python prototype package while still installing a `tarka` console
command.

Do not make PyPI the primary customer install channel. Use it only as:

- a namespace placeholder for TarkaHQ
- a convenience path for Python-heavy users
- a bootstrap mechanism while the native CLI is being built

Package naming:

```text
PyPI package: tarkahq
CLI binary: tarka
```

## Pilot Install: Curl

Use the public GitHub installer for pilots:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | bash
```

Once DNS is configured, wrap the same installer behind:

```bash
curl -fsSL https://install.tarkahq.com/cli | bash
```

The installer:

- creates a user-local virtualenv under `~/.local/share/tarka-cli`
- keeps a source checkout under `~/.local/share/tarka-cli/source`
- installs the Python prototype into the virtualenv in editable mode so
  the Phase 0 `docs/` templates and `scripts/` helpers remain available
- links `tarka` into `~/.local/bin`
- does not require sudo

## GitHub Release Install

After tagging releases, prefer installing immutable artifacts:

```bash
curl -fsSL https://raw.githubusercontent.com/TarkaHQ/tarkahq-cli/main/install.sh | TARKA_CLI_REF=v0.1.0 bash
```

## Homebrew Later

Homebrew is the right customer-quality install path for macOS/Linux once
we finish:

```text
Python dependency resource vendoring
formula install smoke test
```

Target command:

```bash
brew tap TarkaHQ/tap
brew install tarka
```

Formula shape:

```ruby
class Tarka < Formula
  include Language::Python::Virtualenv

  desc "Tarka CLI"
  homepage "https://tarkahq.com"
  url "https://github.com/TarkaHQ/tarkahq-cli/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<release-sha256>"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"tarka", "--help"
  end
end
```

The `TarkaHQ/homebrew-tap` repo exists and the formula points at
`v0.1.0`, but treat Homebrew as experimental until the dependency
resources and install smoke test are finished.

## Recommended Order

1. Now: Python prototype installed by `curl` into a user-local venv.
2. Next: native single-binary CLI released from GitHub.
3. Pilot: `curl` installer that downloads the native binary for the
   user's OS/architecture.
4. Public/customer polish: Homebrew tap.
5. Optional Python-native path: `pipx install tarkahq`.

## Native CLI Target

The durable CLI should be built as a real binary, not a Python app.

Recommended implementation:

```text
Go or Rust
single binary per platform
GitHub Releases for darwin/linux arm64/amd64
curl installer downloads the right binary
Homebrew formula installs from release tarballs
config remains ~/.config/tarka/config.json
command remains tarka
```

The current Python CLI should be treated as the command contract
prototype. Once the command surface settles, port the implementation to
the native CLI and keep behavior compatible:

```text
tarka init
tarka status
tarka models
tarka chat
tarka training init/check/run/monitor/artifacts
```
