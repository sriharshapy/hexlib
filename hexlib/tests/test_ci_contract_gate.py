# hexlib/tests/test_ci_contract_gate.py
"""Exercises the ACTUAL inline script from .github/workflows/ci.yml's
kernel-contract job, extracted from the YAML itself so this test cannot drift
from what CI really runs.

Finding 2: `failed = False` was only ever set inside the `for d in ...` loop,
so an empty (or absent) kernels/ directory made the job exit 0 -- a check that
found nothing to check reported success. That is CI's only enforcement of the
local-results contract, so a PR that deletes kernels/rmsnorm_fp16, a bad merge
that drops it, or a checkout that fails to materialize it, was reported as
fully compliant.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _extract_contract_script() -> str:
    """Pull the heredoc'd Python out of the 'Check every kernel directory'
    step, exactly as it will run in CI. Avoids a PyYAML dependency (not
    installed in the offline CI job) by locating the heredoc directly in the
    raw file text -- the block is a `python - <<'PY' ... PY` heredoc with a
    fixed indentation, so this is a plain string search, not a YAML parse.
    """
    text = CI_YAML.read_text(encoding="utf-8")
    start = text.index("python - <<'PY'") + len("python - <<'PY'")
    end = text.index("\n          PY", start)
    block = text[start:end]
    # Strip the step's common leading indentation (10 spaces in this file).
    lines = [ln[10:] if ln.startswith(" " * 10) else ln for ln in block.splitlines()]
    return "\n".join(lines)


def _run_script(cwd: pathlib.Path) -> subprocess.CompletedProcess:
    script = _extract_contract_script()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def test_empty_kernels_directory_fails_the_gate(tmp_path):
    """A check that found nothing to check must never report success."""
    (tmp_path / "kernels").mkdir()
    result = _run_script(tmp_path)
    assert result.returncode != 0
    assert "no kernel directories" in (result.stdout + result.stderr)


def test_missing_kernels_directory_fails_the_gate(tmp_path):
    """`Path('kernels').glob('*/')` yields nothing rather than raising when the
    directory itself does not exist -- so this must be checked explicitly."""
    assert not (tmp_path / "kernels").exists()
    result = _run_script(tmp_path)
    assert result.returncode != 0


def test_a_real_valid_kernel_directory_passes_the_gate(tmp_path):
    """Sanity check: the fix must not turn a real, compliant kernel red."""
    dst = tmp_path / "kernels" / "rmsnorm_fp16"
    dst.parent.mkdir()
    shutil.copytree(REPO_ROOT / "kernels" / "rmsnorm_fp16", dst)
    result = _run_script(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked 1 kernel directories" in result.stdout
