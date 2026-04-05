# Tokenizer Lane

This lane is separate from the architecture lane. It exists so tokenizer experiments can proceed in parallel without blocking model-side work.

## Objective

- Determine whether tokenizer changes can recover delivered `val_bpb` faster than model-only tuning.
- Keep byte accounting strict so any tokenizer delta is real, not a measurement artifact.
- Prefer cheap profiling and small re-export tests before spending time on large tokenizer sweeps.
- Keep two explicit sub-lanes:
  - `competitive lane`: standard BPE around `2048-4096`
  - `novelty lane`: morphology-aware or meaning-aware tokenization after the competitive lane is benchmarked

## First-wave experiments

1. `Control`
   - Re-run the current `sp1024` tokenizer family as the baseline control for every sweep.
   - Purpose: keep the tokenizer comparison anchored.

2. `Competitive BPE vocab sweep`
   - Compare standard BPE candidates at `1024`, `2048`, and `4096`.
   - Purpose: test the current field-consensus sweet spot before more novel tokenization ideas.
   - Important coupling:
     - evaluate these on top of the strongest non-recurrent stack first, not just the baseline control
     - only pivot the main tokenizer lane toward recurrence if a recurrent delivery path proves competitive

3. `Bytes-per-token control`
   - Measure bytes-per-token and token-length distribution on the fixed validation documents before launching full model sweeps.
   - Purpose: keep the tokenizer lane grounded in the metric the challenge actually scores.

4. `Tokenizer family comparison`
   - Compare SentencePiece BPE against at least one alternative tokenizer configuration on the same docs cache.
   - Purpose: learn whether the current BPE family is already near the best fit for FineWeb.
   - Current priority:
     - keep this secondary to the standard-BPE vocab sweep, not ahead of it.

5. `Corpus profiling`
   - Measure byte distribution, average document length, token-length distribution, and boilerplate density from `docs_selected.jsonl`.
   - Purpose: choose tokenizer settings from the actual corpus mix and feed the document-packing lane.

6. `Model/tokenizer co-design`
   - First test whether a slightly larger vocab wins on the strong non-recurrent stack.
   - If `SP-4096` becomes artifact-limited, test factored embeddings or a low-rank output residual only after the plain larger-vocab path is measured.
   - After structural architecture changes free bytes, test whether a slightly larger vocab plus a smaller or more shared model wins on delivered BPB.
   - Purpose: avoid optimizing tokenizer size in isolation.

7. `Novelty tokenizer comparison`
   - Compare a morphology-aware tokenizer against the best standard-BPE control.
   - For Morfessor-seeded SentencePiece specifically, do not blindly protect raw morphemes as `user_defined_symbols`; first screen them for substring bleed inside unrelated words.
   - Purpose: keep the more theoretical tokenizer work explicit and measurable instead of silently replacing the competitive path.

8. `Offline teacher-assisted scoring only`
   - If we use a stronger external model in this lane, restrict it to offline corpus scoring, curriculum ordering, or tokenizer evaluation.
   - Purpose: stay on the safe side of the competition rules and avoid teacher-dependent training or evaluation paths.

## Risks

- Tokenizer edits can create fake wins if byte accounting is wrong, so every run must use the same BPB calculator as the trainers.
- Re-exporting shards is slow, so only promote tokenizer settings that show signal in profiling or small-scale re-export tests.
- Larger vocabularies consume artifact budget quickly, so tokenizer work must stay coupled to the 16MB limit.
- Larger vocabularies may also force model-side delivery changes like fp16 embedding passthrough or factorization, so tokenizer comparisons should log those choices explicitly.
- Novel tokenizer ideas can absorb a lot of iteration time; they should not block the standard-BPE `2048-4096` path.

## Run logging

- Log every run in `research/iterations/run_index.jsonl` with the exact command, host, dataset variant, tokenizer config, and metric result.
- Use one run per mini per tokenizer modification while the machines are independent.
- Include the shard count, tokenizer path, vocab size, and whether the run is a control or candidate.
- Keep longer notes in the linked iteration archive folder, not in the JSONL row.

## Cluster-ready sweep spec

- The first concrete Mini sweep spec is [tokenizer_candidate_mlx_sweep.json](/home/zaytor/transformer_research/parameter-golf/research/iterations/archive/2026/iter_20260319_102331_tokenizer-lane-baseline-sweep/tokenizer_candidate_mlx_sweep.json).
- Fill in the Mini-local dataset and tokenizer paths once the candidate exports exist on-node.
- Then prepare or launch with `tools/prepare_mlx_sweep.py`.
