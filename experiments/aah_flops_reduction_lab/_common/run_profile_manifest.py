"""Run selected rows from a lab profiling manifest.

This is intentionally small and boring: it executes the explicit commands
emitted by make_profile_manifest.py, records one status row per profile, and
skips outputs that already exist when requested.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_status(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--name-regex", default="")
    parser.add_argument("--only-ready", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()

    name_re = re.compile(args.name_regex) if args.name_regex else None
    rows = load_rows(Path(args.manifest))
    selected = []
    for row in rows:
        if args.only_ready and row.get("status") not in (None, "ready"):
            continue
        if name_re and not name_re.search(row["name"]):
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    append_status(
        Path(args.status),
        {
            "event": "queue_start",
            "manifest": args.manifest,
            "selected": [row["name"] for row in selected],
            "time": time.time(),
        },
    )

    for row in selected:
        output = Path(row["output"])
        if args.skip_existing and output.exists():
            append_status(
                Path(args.status),
                {
                    "event": "skip_existing",
                    "name": row["name"],
                    "output": row["output"],
                    "time": time.time(),
                },
            )
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        append_status(
            Path(args.status),
            {
                "event": "profile_start",
                "name": row["name"],
                "command": row["command"],
                "output": row["output"],
                "time": start,
            },
        )
        result = subprocess.run(row["command"], cwd=args.cwd)
        end = time.time()
        append_status(
            Path(args.status),
            {
                "duration_sec": end - start,
                "event": "profile_finish",
                "name": row["name"],
                "output": row["output"],
                "returncode": result.returncode,
                "time": end,
            },
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    append_status(Path(args.status), {"event": "queue_finish", "time": time.time()})


if __name__ == "__main__":
    main()
