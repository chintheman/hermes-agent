"""node-gyp builds against the Hermes-managed Node's own headers.

node-pty has no Linux prebuild, so ``npm install`` ends in a node-gyp rebuild.
Left to itself node-gyp downloads a headers tarball from nodejs.org through
its bundled undici client, which crashes on a connection that closes while
the download is paused (nodejs/undici#5360) -- the install E2E lost that race
on ~40% of scheduled runs. The managed Node tarball already ships the headers,
so install.sh must hand node-gyp ``npm_config_nodedir`` pointing at it, and
ONLY at it: a guessed nodedir for any other Node breaks node-gyp outright.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

NODE_STUB = "#!/bin/sh\necho v26.0.0\n"
# Records the cwd and the nodedir npm would hand node-gyp, one call per line.
NPM_STUB = """#!/bin/sh
if [ "${1:-}" = "--version" ]; then
    echo 12.0.0
    exit 0
fi
printf '%s\\t%s\\n' "$PWD" "${npm_config_nodedir-<unset>}" >> "$NPM_CALLS"
exit 0
"""


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_node_deps_stage(
    tmp_path: Path,
    *,
    managed_node: bool,
    with_headers: bool = True,
    via_symlink: bool = False,
    preset_nodedir: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    install_dir = tmp_path / "install"
    hermes_home = tmp_path / "home"
    npm_calls = tmp_path / "npm-calls"
    (install_dir / "ui-tui").mkdir(parents=True)
    (install_dir / "package.json").write_text('{"private":true}\n', encoding="utf-8")
    (install_dir / "ui-tui" / "package.json").write_text('{"private":true}\n', encoding="utf-8")
    (hermes_home / "bin").mkdir(parents=True)
    _write_executable(hermes_home / "bin" / "uv", "#!/bin/sh\necho 'uv probe'\n")

    if managed_node:
        toolchain = hermes_home / "node" / "bin"
        if with_headers:
            (hermes_home / "node" / "include" / "node").mkdir(parents=True)
            (hermes_home / "node" / "include" / "node" / "common.gypi").write_text(
                "{}\n", encoding="utf-8"
            )
    else:
        toolchain = tmp_path / "system-bin"
    toolchain.mkdir(parents=True)
    _write_executable(toolchain / "node", NODE_STUB)
    _write_executable(toolchain / "npm", NPM_STUB)

    path_dir = toolchain
    if via_symlink:
        # The real layout: install.sh links node/npm into the command link
        # dir, which is what ends up on PATH; ~/.hermes/node/bin itself is not.
        path_dir = tmp_path / "link-dir"
        path_dir.mkdir()
        for name in ("node", "npm"):
            (path_dir / name).symlink_to(toolchain / name)

    env = os.environ.copy()
    env.pop("npm_config_nodedir", None)
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_INSTALL_DIR": str(install_dir),
            "NPM_CALLS": str(npm_calls),
            "PATH": f"{path_dir}:{env['PATH']}",
        }
    )
    if preset_nodedir is not None:
        env["npm_config_nodedir"] = preset_nodedir
    proc = subprocess.run(
        [
            "bash",
            str(INSTALL_SH),
            "--stage",
            "node-deps",
            "--json",
            "--skip-browser",
            "--skip-computer-use",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = npm_calls.read_text(encoding="utf-8").splitlines()
    return proc, hermes_home, calls


def _nodedirs(calls: list[str]) -> set[str]:
    return {line.split("\t", 1)[1] for line in calls}


def test_managed_node_gets_its_own_headers(tmp_path: Path) -> None:
    proc, hermes_home, calls = _run_node_deps_stage(tmp_path, managed_node=True)

    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2, calls  # root + ui-tui, both see the same nodedir
    assert _nodedirs(calls) == {str(hermes_home / "node")}
    assert "Hermes-managed Node headers" in proc.stdout


def test_managed_node_reached_through_command_link_still_matches(tmp_path: Path) -> None:
    proc, hermes_home, calls = _run_node_deps_stage(
        tmp_path, managed_node=True, via_symlink=True
    )

    assert proc.returncode == 0, proc.stderr
    assert _nodedirs(calls) == {str(hermes_home / "node")}


def test_system_node_is_never_given_a_guessed_nodedir(tmp_path: Path) -> None:
    proc, _, calls = _run_node_deps_stage(tmp_path, managed_node=False)

    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2, calls
    assert _nodedirs(calls) == {"<unset>"}
    assert "Hermes-managed Node headers" not in proc.stdout


def test_managed_node_without_headers_falls_back_to_node_gyp_download(tmp_path: Path) -> None:
    proc, _, calls = _run_node_deps_stage(tmp_path, managed_node=True, with_headers=False)

    assert proc.returncode == 0, proc.stderr
    assert _nodedirs(calls) == {"<unset>"}


def test_callers_own_nodedir_wins(tmp_path: Path) -> None:
    proc, _, calls = _run_node_deps_stage(
        tmp_path, managed_node=True, preset_nodedir="/opt/custom-node"
    )

    assert proc.returncode == 0, proc.stderr
    assert _nodedirs(calls) == {"/opt/custom-node"}
