"""Run an isolated backend experiment with an explicit transformer module."""

import argparse
import os
import sys


MODULES = {
    "pure": "experiments.backend_realized_local_attention._common.pure_backend_transformer",
    "aah": "experiments.backend_realized_local_attention._common.aah_backend_transformer",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", choices=sorted(MODULES), required=True)
    args = parser.parse_args()

    os.environ["AAH_BACKEND_TRANSFORMER_MODULE"] = MODULES[args.module]
    from experiments.backend_realized_local_attention._common import train_backend

    sys.argv = ["train_backend.py", "--config", args.config]
    train_backend.main()


if __name__ == "__main__":
    main()
