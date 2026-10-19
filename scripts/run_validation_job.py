#!/usr/bin/env python3
"""Execute one JSONL validation job, with resumable completion markers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_job(path: Path, index: int):
    if index < 0:
        raise ValueError("Job index must be non-negative.")
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index == index:
                job = json.loads(line)
                argv = job.get("argv")
                if not isinstance(argv, list) or not argv or not all(
                    isinstance(item, str) for item in argv
                ):
                    raise ValueError(f"Job {index} has an invalid argv list.")
                return job
    raise IndexError(f"Job index {index} is outside {path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Zero-based job index; defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument("--status-dir", default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_file = Path(args.job_file).resolve()
    index = args.index
    if index is None:
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_index is None:
            raise ValueError("Provide --index or set SLURM_ARRAY_TASK_ID.")
        index = int(slurm_index)

    job = _load_job(job_file, index)
    status_dir = (
        Path(args.status_dir).resolve()
        if args.status_dir
        else job_file.with_suffix(job_file.suffix + ".status")
    )
    done_path = status_dir / f"{index:06d}.done.json"
    if done_path.exists() and not args.rerun:
        print(f"SKIP completed job {index}: {job.get('id')}")
        return

    command = job["argv"]
    print(f"RUN job {index}: {job.get('id')}")
    print(shlex.join(command))
    if args.dry_run:
        return

    started_at = datetime.now(timezone.utc).isoformat()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    completed_at = datetime.now(timezone.utc).isoformat()
    status_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "index": index,
        "id": job.get("id"),
        "job_file": str(job_file),
        "argv": command,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    temporary_path = done_path.with_suffix(done_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(done_path)


if __name__ == "__main__":
    main()
