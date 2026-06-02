"""Copy one exported static plan to every lab variant that requires a plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.aah_flops_reduction_lab._common.naming import RUN_ROOT, VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--root", default=RUN_ROOT)
    args = parser.parse_args()

    with open(args.source, "r") as f:
        plan = json.load(f)

    root = Path(args.root)
    out_dir = root / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in VARIANTS:
        if not variant.requires_plan:
            continue
        out = out_dir / f"{variant.name}.json"
        with open(out, "w") as f:
            json.dump(plan, f, indent=2, sort_keys=True)
        written.append(str(out))

    print(f"copied_plan_to {len(written)} required variant paths")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
