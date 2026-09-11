#!/usr/bin/env python3
"""
fetch_task.py -- download the result of a Tidy3D task that finished on the
server but did not reach this machine.

    python fetch_task.py <task_id> <destination.hdf5>
    python fetch_task.py --list                 # recent tasks and their ids

This downloads a stored result. It does not resubmit anything, so it costs no
credits (a task with status "success" has already been billed).

WHEN YOU NEED THIS
------------------
Finishing the solve and downloading the result are two separate steps, and a
network interruption during a large (about 300 MB) download can fail the
second. `simulation.run` retries the download automatically. If it still
cannot complete, it prints the task id and the command to run here. After the
download, re-run the original script: it will find the file and skip that
simulation.

FINDING A TASK ID
-----------------
It is in the console output of the original run, on the line that reads

    Created task 'paired_unbound' with resource_id 'fdve-...'

or in the workbench URL printed just below it. `--list` also shows recent
tasks.
"""

from __future__ import annotations

import argparse
import os
import sys


def fetch(task_id: str, path: str, verbose: bool = True) -> bool:
    from simulation import hdf5_looks_complete, load_task

    if hdf5_looks_complete(path):
        print(f"  {path} already looks complete. Nothing to do.")
        print("  (Delete it first if you want to force a fresh download.)")
        return True

    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"  {path} exists but is short or unreadable "
              f"({size / 1e6:.1f} MB), probably from an interrupted")
        print("  download. Removing it so the download starts clean.")
        os.remove(path)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    print(f"  fetching {task_id}")
    print(f"       into {path}")
    print("  This is a download of an already-billed result. Nothing is "
          "resubmitted.")
    load_task(task_id, path=path, verbose=verbose)

    if hdf5_looks_complete(path):
        print(f"\n  recovered: {os.path.getsize(path) / 1e6:.1f} MB")
        print("  Re-run whatever you were running. It will find this file and")
        print("  skip that simulation.")
        return True
    print("\n  The download completed but the file still looks wrong.")
    print("  Try again when the connection is stable.")
    return False


def list_tasks(limit: int = 20) -> None:
    from tidy3d import web

    try:
        folder = web.core.task_core.Folder.get("default")
        tasks = folder.list_tasks()
    except Exception as exc:
        raise SystemExit(
            f"Could not list tasks: {exc}\n"
            "Read the id from the console output of the original run instead, "
            "on the line beginning \"Created task ... with resource_id\"."
        )
    print(f"  {'task id':<40} {'name':<24} status")
    for t in list(tasks)[:limit]:
        tid = getattr(t, "task_id", "?")
        name = getattr(t, "task_name", "?")
        status = getattr(t, "status", "?")
        print(f"  {tid:<40} {str(name):<24} {status}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_id", nargs="?", help="fdve-... from the run log")
    ap.add_argument("path", nargs="?", help="where to write the .hdf5")
    ap.add_argument("--list", action="store_true",
                    help="show recent tasks and their ids")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_tasks()
        raise SystemExit(0)
    if not (args.task_id and args.path):
        ap.error("give a task id and a destination, or use --list")

    ok = fetch(args.task_id, args.path, verbose=not args.quiet)
    raise SystemExit(0 if ok else 1)
