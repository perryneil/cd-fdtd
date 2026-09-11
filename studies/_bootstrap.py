"""
_bootstrap.py -- make the scripts in studies/ runnable from anywhere.

The simulation package lives in `cdfdtd/`, and its scripts read and write
`data/...` paths relative to that directory. The scripts in `studies/` sit one
level away, so they locate the package, add it to `sys.path` and change into it
before importing anything.

Call `setup()` first, before any project import.
"""
from __future__ import annotations

import os
import sys

MARKERS = ("simulation.py", "config.py", "observables.py", "tissue.py")


def find_package(start: str | None = None) -> str:
    """Locate the directory holding the project modules."""
    here = os.path.dirname(os.path.abspath(start or __file__))
    candidates = []
    for base in (here, os.path.dirname(here), os.getcwd(),
                 os.path.dirname(os.path.abspath(os.getcwd()))):
        candidates += [base, os.path.join(base, "cdfdtd")]
    seen = set()
    for c in candidates:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if all(os.path.isfile(os.path.join(c, m)) for m in MARKERS):
            return c
    raise SystemExit(
        "Could not find the project package.\n"
        "  Looked for simulation.py, config.py, observables.py and tissue.py in:\n"
        + "\n".join(f"    {c}" for c in sorted(seen))
        + "\n\n  Run this from inside the repository, for example:\n"
          "    cd path/to/this/repository\n"
          "    python studies/tissue_signal.py\n"
    )


def setup(verbose: bool = True) -> str:
    """Put the package on sys.path and chdir into it.

    The chdir matters: every script in this project reads and writes
    `data/<name>/` relative to the package directory, where completed runs are
    stored. Without it a script would create a new, empty `data/` folder under
    `studies/` and could not find (or would repeat) runs that already exist.
    """
    pkg = find_package()
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    if os.path.abspath(os.getcwd()) != pkg:
        os.chdir(pkg)
        if verbose:
            print(f"  working directory -> {pkg}")
    return pkg
