| Method | MMLU | MMLU-Pro | GPQA-Diamond | ARC-Challenge | HellaSwag | TriviaQA | GSM8K | HumanEval | MBPP | C-Eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full attention | 70.9 | 28.5 | 34.8 | 86.3 | 44.3 | 37.5 | 13.3 | 34.4 | 3.1 | 71.5 |
| Full adaptive | 71.1 | 27.7 | 34.3 | 86.3 | 44.3 | 37.5 | 13.3 | 34.4 | 3.1 | 71.5 |
| Deep practical reuse | 70.9 | 27.7 | 34.3 | 86.3 | 44.3 | 37.5 | 13.3 | 34.4 | 3.1 | 71.5 |
| Shallow freeze | 70.7 | 27.7 | 33.8 | 86.3 | 44.3 | 37.5 | 14.8 | 34.4 | 3.1 | 71.5 |
| Grouping off | 70.9 | 28.5 | 35.4 | 86.3 | 44.5 | 37.5 | 13.3 | 34.4 | 3.1 | 71.9 |

Scores are percentages. We evaluated Qwen3-4B-Base with the same tokenizer, context length 4096, bf16 inference, and identical fixed evaluation subsets across all methods. Multiple-choice tasks were scored by normalized log-likelihood over answer choices. Open-ended QA/math tasks used greedy decoding with temperature 0 and exact-match style answer extraction. Code tasks used greedy decoding and pass@1 execution against the provided unit tests. AAH variants used the pretrained Qwen3 backbone with the corresponding adapted AAH controller/topology parameters loaded; the full-attention row uses the original Qwen3 attention path. This is an internal capped-subset compatibility smoke test, not an official full benchmark report. The transfer/load map, displayed subset counts, recorded adapter hashes, and remaining reproducibility gaps are documented in `../downstream_provenance_manifest.md`.
