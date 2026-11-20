"""`python scripts/<name>.py` must find the jncm package without an editable install.

2026-11-20: every eval job on FIR died with `ModuleNotFoundError: No module named 'jncm'`
at scripts/run_eval.py line 45. Python puts the script's own directory (scripts/) on
sys.path, not the repo root, and the cluster venv has no editable install. The CPU tests
never saw it: they import `scripts.run_eval` from the repo root, and the dev venv has
jncm installed. The probe below removes both (the editable finder and the repo path
entry) before it loads the script, so it fails exactly like the cluster did.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

SCRIPTS = ["scripts/run_eval.py", "scripts/eval_data_row.py"]

PROBE = textwrap.dedent(
    """
    import importlib.util
    import runpy
    import sys
    from pathlib import Path

    script, root = sys.argv[1], Path(sys.argv[2]).resolve()
    # setuptools' editable finder is a class on sys.meta_path, so ask the object itself
    sys.meta_path[:] = [
        f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "").lower()
    ]
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != root]
    for name in [n for n in sys.modules if n == "jncm" or n.startswith("jncm.")]:
        del sys.modules[name]
    assert importlib.util.find_spec("jncm") is None, "probe failed to hide jncm"

    runpy.run_path(script, run_name="probe")
    import jncm
    assert Path(jncm.__file__).resolve().parent.parent == root, jncm.__file__
    print("PROBE_OK")
    """
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_finds_jncm_without_an_install(script: str) -> None:
    root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(root / script), str(root)],
        capture_output=True,
        text=True,
        cwd=str(root.parent),
        timeout=300,
    )

    assert "PROBE_OK" in result.stdout, (script, result.stdout[-500:], result.stderr[-800:])
