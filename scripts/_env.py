"""Entry guard shared by the scripts.

Running `python scripts/whatever.py` with the system interpreter fails with a
bare `ModuleNotFoundError: No module named 'tokyoline'`, which says nothing about
the cause. The package and its dependencies are installed only into the
project's virtual environment, so the fix is always the same and the error should
say so.

Two things happen here:

1. `src/` is put on `sys.path`, so the scripts work from a checkout even when the
   package has not been pip-installed.
2. If the import still fails -- because a dependency such as pandas or scipy is
   missing from whichever interpreter is running -- the failure is turned into an
   instruction naming the exact command to use.

Import and call `ensure()` as the first statement of every script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if sys.platform == "win32":
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
    ACTIVATE = r".venv\Scripts\Activate.ps1"
else:
    VENV_PY = ROOT / ".venv" / "bin" / "python"
    ACTIVATE = "source .venv/bin/activate"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def ensure(extra: tuple[str, ...] = (), *, streamlit_app: str | None = None) -> None:
    """Make the package importable, or explain clearly why it is not.

    Parameters
    ----------
    extra : additional module names to verify, beyond the package itself.
        Used by the GUI, which needs plotly -- a dependency the system
        interpreter may well be missing even when streamlit itself is present.
    streamlit_app : when set, the suggested command becomes a streamlit
        invocation rather than a plain script run.

    The streamlit case is worth handling separately because its failure mode is
    genuinely confusing: `python -m streamlit run app.py` succeeds in launching
    the server if streamlit happens to be installed system-wide, and only then
    fails inside the app on some other import. The traceback appears in the
    browser, points at a line in app.py, and gives no hint that the interpreter
    is the problem.
    """
    if SRC.is_dir() and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    missing = None
    try:
        import tokyoline  # noqa: F401
        for name in extra:
            __import__(name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "a dependency"

    if missing is None:
        return

    running = Path(sys.executable)
    in_venv = VENV_PY.exists() and running.resolve() == VENV_PY.resolve()

    if streamlit_app:
        command = f"{_relative(VENV_PY)} -m streamlit run {streamlit_app}"
    else:
        command = f"{_relative(VENV_PY)} {_relative(Path(sys.argv[0]))}"

    lines = [
        "",
        f"  Cannot import {missing!r}.",
        "",
    ]

    if not in_venv and VENV_PY.exists():
        lines += [
            f"  You are running:  {running}",
            f"  but this project's dependencies are installed in its virtual",
            f"  environment only.",
            "",
            "  Run it with the venv interpreter:",
            "",
            f"      {command}",
            "",
            "  or activate the environment first:",
            "",
            f"      {ACTIVATE}",
            "",
        ]
    elif not VENV_PY.exists():
        lines += [
            "  No virtual environment found at .venv. Create one and install",
            "  the project:",
            "",
            "      python -m venv .venv",
            f"      {_relative(VENV_PY)} -m pip install -e .",
            "",
        ]
    else:
        lines += [
            "  The virtual environment is active but the package is incomplete.",
            "  Reinstall it:",
            "",
            f"      {_relative(VENV_PY)} -m pip install -e .",
            "",
        ]

    message = "\n".join(lines)
    print(message, file=sys.stderr)

    # Under streamlit the process is the server, and raising SystemExit here
    # would leave the browser showing a bare traceback with the real
    # explanation buried in the terminal. Put it on the page instead.
    if streamlit_app and "streamlit" in sys.modules:
        try:
            import streamlit as st

            st.error("Wrong Python interpreter — dependencies are missing.")
            st.code(command, language="bash")
            st.caption(
                f"Cannot import '{missing}'. You are running {running}, but this "
                "project's dependencies are installed in its .venv only."
            )
            st.stop()
        except ImportError:
            pass

    raise SystemExit(1)
