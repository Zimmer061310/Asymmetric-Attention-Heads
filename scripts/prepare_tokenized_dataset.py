#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_texts_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def encode_text(tokenizer, text):
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].squeeze(0).to(dtype=torch.long).cpu()


def main():
    parser = argparse.ArgumentParser(description="Prepare an offline tokenized dataset for AAH training.")
    parser.add_argument("--output", required=True, help="Output .pt path.")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or local tokenizer path.")
    parser.add_argument("--dataset", default="wikitext", help="Hugging Face dataset builder name.")
    parser.add_argument("--dataset-name", default="wikitext-2-raw-v1", help="Hugging Face dataset config name.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--train-text-file", default="", help="Optional local train text file; bypasses HF dataset loading.")
    parser.add_argument("--val-text-file", default="", help="Optional local validation text file; bypasses HF dataset loading.")
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-val-tokens", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.train_text_file and args.val_text_file:
        train_text = load_texts_from_file(args.train_text_file)
        val_text = load_texts_from_file(args.val_text_file)
        dataset_source = {
            "type": "text_files",
            "train_text_file": os.path.abspath(args.train_text_file),
            "val_text_file": os.path.abspath(args.val_text_file),
        }
    else:
        ds = load_dataset(args.dataset, args.dataset_name)
        train_text = "\n\n".join(ds[args.train_split]["text"])
        val_text = "\n\n".join(ds[args.val_split]["text"])
        dataset_source = {
            "type": "huggingface",
            "dataset": args.dataset,
            "dataset_name": args.dataset_name,
            "train_split": args.train_split,
            "val_split": args.val_split,
        }

    train_tokens = encode_text(tokenizer, train_text)
    val_tokens = encode_text(tokenizer, val_text)
    if args.max_train_tokens > 0:
        train_tokens = train_tokens[: args.max_train_tokens]
    if args.max_val_tokens > 0:
        val_tokens = val_tokens[: args.max_val_tokens]

    payload = {
        "train": train_tokens,
        "validation": val_tokens,
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer": args.tokenizer,
        "source": dataset_source,
        "num_train_tokens": int(train_tokens.numel()),
        "num_validation_tokens": int(val_tokens.numel()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    meta_path = output.with_suffix(output.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in payload.items() if k not in {"train", "validation"}},
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote {output}")
    print(f"wrote {meta_path}")
    print(f"train_tokens={train_tokens.numel()} validation_tokens={val_tokens.numel()} vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
