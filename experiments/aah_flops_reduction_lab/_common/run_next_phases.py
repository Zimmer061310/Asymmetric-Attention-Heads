"""Continue FLOPs lab profiling phases after an earlier queue finishes.

The intended remote flow is:

1. Wait for the already-running H2 queue to finish cleanly.
2. Export one static AAH plan and copy it to all plan-required variants.
3. Run plan-required Nsight profile phases one at a time.

This script does not release rented instances. It only writes logs/status files
under the lab result root and exits on the first failed prerequisite or phase.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


PHASES = (
    ("h1_static_compiled_plan", r"static-plan"),
    ("h4_fixed_plan_granularity", r"fixed|slow-update"),
    ("h3_noscatter_prototype", r"noscatter"),
)


def read_status_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("time", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()


def wait_for_clean_finish(path: Path, poll_sec: int, status_path: Path) -> None:
    append_status(status_path, {"event": "wait_start", "wait_for": str(path)})
    while True:
        events = read_status_events(path)
        failed = [
            row
            for row in events
            if row.get("event") == "profile_finish" and int(row.get("returncode", 0)) != 0
        ]
        if failed:
            append_status(status_path, {"event": "wait_failed", "failed": failed[-1]})
            raise SystemExit(f"prior queue failed: {failed[-1]}")
        if any(row.get("event") == "queue_finish" for row in events):
            append_status(status_path, {"event": "wait_finish", "wait_for": str(path)})
            return
        time.sleep(int(poll_sec))


def run_checked(cmd: list[str], cwd: Path, status_path: Path, event: str) -> None:
    append_status(status_path, {"command": cmd, "event": f"{event}_start"})
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    append_status(
        status_path,
        {
            "duration_sec": time.time() - start,
            "event": f"{event}_finish",
            "returncode": result.returncode,
        },
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="paper_results/aah_flops_reduction_lab")
    parser.add_argument("--manifest", default="paper_results/aah_flops_reduction_lab/profile_manifest_pro6000.jsonl")
    parser.add_argument("--h2-status", default="paper_results/aah_flops_reduction_lab/profile_status_h2_pro6000.jsonl")
    parser.add_argument("--status", default="paper_results/aah_flops_reduction_lab/profile_status_autophases_pro6000.jsonl")
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    root = Path(args.root)
    status_path = Path(args.status)

    wait_for_clean_finish(Path(args.h2_status), args.poll_sec, status_path)

    export_cmd = [
        "bash",
        "experiments/aah_flops_reduction_lab/h1_static_compiled_plan/scripts/export_static_plan.sh",
    ]
    run_checked(export_cmd, cwd, status_path, "export_static_plan")

    for phase_name, name_regex in PHASES:
        phase_status = root / f"profile_status_{phase_name}_pro6000.jsonl"
        cmd = [
            "python",
            "-u",
            "-m",
            "experiments.aah_flops_reduction_lab._common.run_profile_manifest",
            "--manifest",
            args.manifest,
            "--status",
            str(phase_status),
            "--name-regex",
            name_regex,
            "--skip-existing",
            "--cwd",
            str(cwd),
        ]
        append_status(
            status_path,
            {
                "event": "phase_dispatch",
                "phase": phase_name,
                "name_regex": name_regex,
                "phase_status": str(phase_status),
            },
        )
        run_checked(cmd, cwd, status_path, phase_name)

    append_status(status_path, {"event": "all_phases_finish"})


if __name__ == "__main__":
    main()
