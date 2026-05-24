# Qwen3-4B Downstream Compatibility Provenance Manifest

This manifest addresses the release-review concern that the Qwen3 downstream
table is not independently reproducible without a transfer/load map, subset
definition, scoring details, and artifact hashes.

The current table should be described as an internal capped-subset
compatibility smoke test, not an official full benchmark report. Exact
percentages should not be cited as benchmark results or used for external
model comparisons until the remaining manifest gaps are resolved.

## Base Model

- Model repo: `Qwen/Qwen3-4B-Base`
- Resolved Hugging Face revision checked locally: `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- License string checked from the Hugging Face model card metadata:
  `apache-2.0`
- Loading path: `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, attn_implementation="eager")`
- Tokenizer path: `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- Context length used by the compatibility runner: `4096`
- Precision requested by the workflow: `bf16`

## AAH Transfer And Load Map

The full-attention baseline uses the original Qwen3 attention path.

For AAH rows, the workflow:

1. Loads the pretrained Qwen3-4B-Base model.
2. Patches attention modules through `scripts/qwen3_aah_patch.py::patch_model_attention`.
3. Selects regime settings through `config_from_regime`:
   - `grouping_off`: `min_group_size=1`, `max_depth=0`.
   - `full_adaptive`: `min_group_size=2`, `max_depth=4`.
   - `shallow_freeze`: `min_group_size=2`, `max_depth=1`, `freeze_learned_topology=True`.
   - `deep_practical_reuse`: `min_group_size=2`, `max_depth=4`, `reuse_group_hierarchy=True`.
4. Loads adapter state, when present, with `load_aah_adapter`.
5. `load_aah_adapter` loads only state-dict keys containing `.aah_state.`; unexpected non-AAH keys are treated as hard errors.

Adapter `.pt` files are intentionally not tracked in Git. The benchmark CSV
records the evaluated adapter hash in `checkpoint_sha256`; those files must be
published separately or regenerated before claiming independent reproduction.

## Evaluated Rows And Recorded Hashes

These hashes are copied from
`paper_results/qwen3_4b_aah/benchmarks/benchmark_results_by_task.csv`.

| Method | Recorded `checkpoint_sha256` |
|---|---|
| `qwen3_4b_full_attention_baseline` | `ab512c42eb4622545690d9e76c13bcb068c3faaadb66010c85f983f906ab80bc` |
| `qwen3_4b_full_adaptive` | `5c305c0ee17ab41b592b34bfbe44646f9ada1746de3f383d06599639ab74f491` |
| `qwen3_4b_deep_practical_reuse` | `23964ed40ec905bd234fb4393694bc1090871b00a1fdcba89dccd9456ed4e855` |
| `qwen3_4b_shallow_freeze` | `cffaf5e81cf9b410a3746fc502d28bd2c108d137fe10423daa46e66b81c02716` |
| `qwen3_4b_grouping_off` | `6bd207e74192eb52cde69a9bae36009e793ae20e15b01651b18081b43cd55650` |

The baseline value is a synthetic run identifier derived from
`model:regime:base`, not a downloaded Qwen weight-file hash. AAH row values are
intended to be adapter-file hashes when adapter files are available during
evaluation.

## Displayed Paper Subsets

The displayed paper table keeps only nonzero, successful tasks from the capped
compatibility run:

| Task | Metric | Displayed examples |
|---|---:|---:|
| MMLU | accuracy | 512 |
| MMLU-Pro | accuracy | 256 |
| GPQA-Diamond | accuracy | 198 |
| ARC-Challenge | accuracy | 256 |
| HellaSwag | accuracy | 512 |
| TriviaQA | exact match | 128 |
| GSM8K | exact match | 128 |
| HumanEval | pass@1 | 32 |
| MBPP | pass@1 | 32 |
| C-Eval | accuracy | 256 |

The runner also attempted additional tasks in raw outputs. Zero-score,
failed, or unavailable tasks are not part of the displayed paper table.

## Scoring Protocol

- Multiple-choice tasks: normalized log-likelihood over answer choices.
- Open-ended QA/math tasks: greedy decoding with temperature 0 and exact-match
  or numeric answer extraction.
- Code tasks: greedy decoding and pass@1 execution against provided unit tests.
- Prompt and loader definitions are implemented in
  `scripts/benchmark_paper_tasks.py`.
- Qwen model loading, patching, adaptation, and benchmark orchestration are in
  `scripts/qwen3_aah_paper.py` and `scripts/run_qwen3_aah_paper.py`.

## Remaining Reproducibility Gaps

The table is still not independently reproducible from this Git repository
alone until these items are published or regenerated:

- exact adapter `.pt` files or public artifact URLs;
- adapter/controller/topology hashes for each AAH row;
- exact dataset revisions for every Hugging Face dataset loaded by the runner;
- exact subset item IDs, not only deterministic first-N counts;
- server-side package versions from the actual run;
- code-task execution environment details.

Until those are available, cite the table only as an internal capped-subset
compatibility check. Do not cite the exact percentages as benchmark results.
