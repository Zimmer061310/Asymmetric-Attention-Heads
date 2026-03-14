#!/usr/bin/env python3
import argparse
import pandas as pd


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="W&B export CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    to_num(
        df,
        [
            "val/loss",
            "val/ppl",
            "perf/tok_s",
            "infer/val_loss",
            "infer/val_ppl",
            "infer/tok_s",
            "aah/flops_ratio",
            "infer/flops_ratio",
        ],
    )

    refs = {
        "v3_final_candidate_E": [
            "aah-v3-full-1b-10k-phase2-E-ema015-r2000-wt2",
            "aah-v3-full-1b-10k-v3-final-wt2",
        ],
        "baseline_10k": ["baseline-1b-10k-wt2"],
        "old_v3_full_10k": ["aah-v3-full-1b-10k-wt2"],
        "previous_night3_10k": ["aah-v3-full-1b-10k-night3-ema015-r4000-wt2"],
    }
    expected_ckpt_substrings = {
        "v3_final_candidate_E": ["phase2-E-ema015-r2000", "aah-v3-final-candidate.pt"],
        "baseline_10k": ["baseline-1b-10k-wt2.pt"],
        "old_v3_full_10k": ["aah-v3-full-1b-10k-wt2.pt"],
        "previous_night3_10k": ["10k-night3-ema015-r4000", "aah-v3-phase1-winner.pt"],
    }

    rows = []
    for label, names in refs.items():
        train = df[df["Name"].isin(names)]
        infer = df[df["Name"].isin([f"{n}-infer" for n in names])]

        train_f = train[train["State"].astype(str).str.lower() == "finished"]
        infer_f = infer[infer["State"].astype(str).str.lower() == "finished"]
        if "infer/checkpoint" in infer_f.columns:
            needles = expected_ckpt_substrings.get(label, [])
            if needles:
                mask = infer_f["infer/checkpoint"].astype(str).apply(
                    lambda s: any(n in s for n in needles)
                )
                if mask.any():
                    infer_f = infer_f[mask]

        t = train_f.iloc[-1] if len(train_f) else None
        if len(infer_f):
            if "infer/val_loss" in infer_f.columns and infer_f["infer/val_loss"].notna().any():
                infer_f = infer_f.sort_values(
                    ["infer/val_loss", "infer/val_ppl", "infer/tok_s"],
                    ascending=[True, True, False],
                )
            i = infer_f.iloc[0]
        else:
            i = None

        flops_col = "infer/flops_ratio" if i is not None and "infer/flops_ratio" in i.index and pd.notna(i["infer/flops_ratio"]) else "aah/flops_ratio"
        rows.append(
            {
                "label": label,
                "train_name": t["Name"] if t is not None else None,
                "train_val_loss": t["val/loss"] if t is not None and "val/loss" in t.index else None,
                "train_val_ppl": t["val/ppl"] if t is not None and "val/ppl" in t.index else None,
                "train_tok_s": t["perf/tok_s"] if t is not None and "perf/tok_s" in t.index else None,
                "infer_name": i["Name"] if i is not None else None,
                "infer_val_loss": i["infer/val_loss"] if i is not None and "infer/val_loss" in i.index else None,
                "infer_val_ppl": i["infer/val_ppl"] if i is not None and "infer/val_ppl" in i.index else None,
                "infer_tok_s": i["infer/tok_s"] if i is not None and "infer/tok_s" in i.index else None,
                "infer_flops_ratio": i[flops_col] if i is not None and flops_col in i.index else None,
                "infer_checkpoint": i["infer/checkpoint"] if i is not None and "infer/checkpoint" in i.index else None,
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
