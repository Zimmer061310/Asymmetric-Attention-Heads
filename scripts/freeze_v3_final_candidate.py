#!/usr/bin/env python3
import os
import shutil


def main():
    src_ckpt = "experiments/aah-v3-full-1b-10k-phase2-E-ema015-r2000-wt2.pt"
    src_cfg = "configs/aah_v3_full_1b_10000_phase2_E.yaml"
    dst_dir = "experiments/final"
    dst_ckpt = os.path.join(dst_dir, "aah-v3-final-candidate.pt")
    dst_cfg = os.path.join(dst_dir, "aah-v3-final-candidate.yaml")

    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt}")
    if not os.path.exists(src_cfg):
        raise FileNotFoundError(f"Source config not found: {src_cfg}")

    shutil.copy2(src_ckpt, dst_ckpt)
    shutil.copy2(src_cfg, dst_cfg)

    print(f"Frozen checkpoint: {dst_ckpt}")
    print(f"Frozen config: {dst_cfg}")


if __name__ == "__main__":
    main()
