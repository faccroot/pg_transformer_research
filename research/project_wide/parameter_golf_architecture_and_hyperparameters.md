# Parameter Golf Baseline: Architecture and Hyperparameter Review

This note covers both starter scripts:

- `train_gpt.py`: the CUDA / distributed PyTorch path.
- `train_gpt_mlx.py`: the Apple Silicon MLX path intended for fast local iteration.

When the code or README states the reason directly, the note treats that as explicit. When it does not, the "why" below is a likely rationale inferred from the challenge constraints and the implementation.

## Constraint-level choices

- Single-file starter scripts under 1500 lines: chosen to keep the baseline readable for newcomers and keep challenge submissions close to the counted artifact format. Common alternatives: a multi-file training stack, custom package layout, code generation, or specialized kernel modules.
- Tokenizer-agnostic evaluation via bits per byte instead of fixed-token loss: chosen because in the 16MB regime the tokenizer and embedding table are part of the compression problem, not just preprocessing. Common alternatives: fixed-tokenizer validation loss, perplexity, byte-level cross-entropy, or compression against a fixed external codec.
- Counted artifact equals code bytes plus compressed model bytes: chosen to reward end-to-end compactness, including serialization strategy. Common alternatives: parameter count only, uncompressed checkpoint size, fp16 checkpoint size, or ignoring code size.
- Published shard format plus fixed validation split: chosen to standardize comparisons and keep local iteration aligned with the challenge baseline. Common alternatives: streaming directly from Hugging Face datasets, custom preprocessing pipelines, or dynamic online shuffling.

## Runtime and bookkeeping knobs

- `DATA_PATH=./data/datasets/fineweb10B_sp1024`: chosen so the default run points at the published 1024-token export immediately. Common alternatives: a different tokenizer export such as `byte260` or `sp4096`, remote mounts, or custom shards.
- `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`: chosen so tokenizer and shard layout match by default. Common alternatives: other SentencePiece models, byte-level tokenizers, or a custom vocabulary trained from the published docs cache.
- `RUN_ID=<uuid>`: chosen so every run is log-addressable without requiring manual naming. Common alternatives: human-only run names, timestamp-only IDs, or an external experiment tracker.
- `SEED=1337`: chosen as a stable deterministic default for baseline comparison. Common alternatives: random seeds per run, seed sweeps, or challenge-specific seed policies.
- `TRAIN_LOG_EVERY=200`: chosen to keep logs informative without overwhelming stdout. Common alternatives: per-step logging, sparse milestone logging, or external metrics dashboards.
- `VAL_LOSS_EVERY=1000` in PyTorch and `0` in MLX: chosen because full validation is expensive, especially on local Macs, while the CUDA path can afford periodic checks. Common alternatives: validate every step range, validate only at the end, or maintain a small proxy validation subset.
- `VAL_BATCH_SIZE=524288`: chosen to keep validation throughput high and reduce loop overhead. Common alternatives: smaller memory-safe validation batches or adaptive batch sizing based on host memory.
- `OUT_DIR=logs` in MLX: chosen to keep local outputs centralized and simple. Common alternatives: timestamped output trees, per-host log roots, or external artifact stores.

## Data pipeline and evaluation decisions

- Manifest-driven cached dataset download: chosen to make the local dataset reproducible and to allow prefix downloads of the published training export. Common alternatives: direct dataset streaming, local preprocessing from raw FineWeb, or ad hoc local shard lists.
- SentencePiece tokenizer requirement for both starter scripts: chosen because the byte-accounting logic is implemented against SentencePiece piece metadata. Common alternatives: byte-level tokenizers, tiktoken, unigram or BPE variants with a different evaluation adapter.
- Full validation split every time validation runs: chosen because the challenge score is defined on the full frozen validation set, so proxy validation can drift. Common alternatives: a smaller dev subset during iteration or mixed periodic small/full validation.
- Token-byte lookup tables (`base_bytes`, leading-space, boundary flags): chosen to compute tokenizer-agnostic bits per byte cheaply at runtime. Common alternatives: decode every token sequence back to text each validation step, offline byte counts, or tokenizer-specific shortcuts.
- Sequential shard streaming with no worker pool and no random sampling: chosen for determinism, simplicity, and minimal code. Common alternatives: shuffled sampling, DataLoader workers, document-level packing, or probabilistic shard mixing.
- Contiguous rank partitioning in `DistributedTokenLoader`: chosen so every rank consumes disjoint spans from the same shared token window without extra synchronization. Common alternatives: separate per-rank streams, random sampling per rank, or pre-sharded rank-local datasets.
- MLX manifest validation of dataset/tokenizer compatibility: chosen to fail fast on silent tokenizer mismatches, which would corrupt `val_bpb`. Common alternatives: trusting filenames, no validation, or hashing both artifacts.

## Core model architecture

- Autoregressive causal transformer objective: chosen because it is the simplest language-model baseline that is directly compatible with the challenge metric and prior NanoGPT-style work. Common alternatives: encoder-decoder language models, masked language modeling, state-space models, retention, or test-time training systems.
- Nine transformer blocks (`NUM_LAYERS=9`): chosen as a small-but-nontrivial depth budget that fits the artifact and time constraints while leaving room for skip structure. Common alternatives: fewer wider blocks, deeper thinner stacks, recurrent depth, or shared-layer loops.
- Width `MODEL_DIM=512`: chosen as a balanced baseline width for a tiny model where attention, MLPs, and embeddings all matter. Common alternatives: narrower models to buy depth, wider models to improve token mixing, or uneven layer widths.
- `NUM_HEADS=8`, giving `HEAD_DIM=64`: chosen because 64-dim heads are conventional, stable, and FlashAttention-friendly. Common alternatives: fewer larger heads, more smaller heads, MQA with one KV head, or head dimensions not tied to 64.
- `NUM_KV_HEADS=4` grouped-query attention: chosen to reduce KV projection parameters and runtime bandwidth while keeping more than one KV group. Common alternatives: full multi-head attention with 8 KV heads, pure multi-query attention with 1 KV head, or more aggressive shared KV structure.
- `VOCAB_SIZE=1024` with the published `sp1024` tokenizer: chosen because small-vocab tokenization dramatically reduces embedding and LM-head footprint in a 16MB artifact regime. Common alternatives: byte-level vocabularies, `sp4096`, larger BPEs, or learned adaptive vocabularies.
- Sequence length `TRAIN_SEQ_LEN=1024`: chosen because it matches the published baseline export and gives enough context to benefit from attention without blowing up local iteration cost. Common alternatives: shorter sequences for speed, longer contexts for compression gains, or curriculum schedules.
- Bias-free linear layers throughout attention and MLP blocks: chosen to save parameters and keep the tiny baseline clean. Common alternatives: learned biases everywhere, biases only in specific projections, or gated affine variants.
- RMSNorm instead of LayerNorm: chosen because RMSNorm is cheaper, simpler, and common in modern efficient transformers. Common alternatives: LayerNorm, ScaleNorm, no normalization, or learned affine RMSNorm variants.
- No learned norm weights in the MLX path and effectively functional RMSNorm in the PyTorch path: chosen to minimize parameters and code complexity. Common alternatives: affine RMSNorm, per-channel gain-only norms, or full LayerNorm affine parameters.
- RMSNorm on token embeddings before entering the stack: chosen to stabilize the initial residual stream and make tied embeddings easier to train. Common alternatives: no embedding norm, learned embedding scale, or post-block-only normalization.
- Per-head Q/K RMSNorm before applying RoPE: chosen to stabilize attention logits in a very small model and make `q_gain` easier to use. Common alternatives: no QK normalization, only query normalization, cosine attention, or scaled initialization alone.
- MLX TurboQuant branch finding on attention geometry: offline projection tests on the untied Turbo checkpoint show that forcing all nine `K` matrices into exact `diag × orthogonal` form is effectively free on capped validation, while forcing all nine `Q` matrices costs about `+0.037` BPB. The practical takeaway is that learned `K` projections already live very close to the rotation-plus-scale geometry assumed by Turbo/Polar-style codecs, while `Q` still uses non-orthogonal structure. That helps explain why `k-only prod` outperformed `q+k prod` in the MLX Turbo sweeps.
- RoPE with `ROPE_BASE=10000`: chosen because RoPE is standard, cheap, and extrapolation-friendly while keeping position handling parameter-free. Common alternatives: learned absolute embeddings, ALiBi, xPos, no position scheme, or different RoPE bases.
- Learnable per-head `QK_GAIN_INIT=1.5`: chosen to let attention logit scale move away from the vanilla `1/sqrt(d)` default if needed, while starting from a stronger-than-neutral signal. Common alternatives: fixed attention scale only, no learnable gain, or deeper attention-temperature parameterization.
- `attn_scale` and `mlp_scale` per feature: chosen to give each block cheap learned residual gating without adding a full gate network. Common alternatives: plain residual adds, scalar layer-level gates, SwiGLU-style gating, or residual adapters.
- `resid_mix` per feature mixing current state with the original embedding stream `x0`: chosen to give every block a cheap learned shortcut to the initial representation, which is attractive in tiny models where information can wash out quickly. Common alternatives: plain residual stacks, learned scalar skip coefficients, HyperConnections, or explicit dense skip wiring.
- Encoder/decoder-style split inside a decoder-only LM (`num_encoder_layers = num_layers // 2`, reversed skip reuse in the second half): chosen to approximate U-Net-like reuse of intermediate features without changing the causal objective. Common alternatives: a plain stack, all-to-all skip connections, recurrent state reuse, or reversible layers.
- `skip_weights` per feature for the cross-half skips: chosen to control how strongly earlier representations re-enter later layers. Common alternatives: fixed unit skip connections, scalar skip strengths, gated skip MLPs, or concatenative skip merges.
- Zero initialization of attention output projections and MLP output projections: chosen to make each block start close to an identity map and reduce early training instability in a compiled, large-batch regime. Common alternatives: standard Xavier/Kaiming init, scaled residual init, DeepNet-style scaling, or learned residual branch rescaling.
- MLP hidden expansion `MLP_MULT=2`: chosen because the parameter budget is tight; the baseline spends relatively more on depth and embeddings than on a wide feed-forward layer. Common alternatives: 4x GPT-style expansion, 3x/8x gated MLPs, low-rank MLPs, or no MLP block at all.
- ReLU squared activation (`relu^2`) instead of GELU or SiLU: chosen because it is cheap, simple, and inherited from modded-nanogpt where it works well in fast training settings. Common alternatives: GELU, SiLU/SwiGLU/GEGLU, tanh-based MLPs, or sparsity-inducing activations.
- Final RMSNorm before logits: chosen to stabilize the output distribution and keep tied-output behavior cleaner. Common alternatives: no final norm, LayerNorm, or learned logit scaling only.
- Tied embeddings by default (`TIE_EMBEDDINGS=1`): chosen because in this challenge embeddings are a large fraction of the parameter budget, so tying saves a lot of bytes. Common alternatives: untied LM heads, factorized heads, adaptive softmax, or low-rank output heads.
- Untied head support only in PyTorch, with a separate `HEAD_LR`: chosen because it is useful for experimentation on CUDA but not necessary for the simple MLX baseline. Common alternatives: no untied path at all or a full mirrored implementation in both scripts.
- `TIED_EMBED_INIT_STD=0.005`: chosen to keep tied embeddings small at init so the shared input/output matrix does not dominate logits too early. Common alternatives: larger GPT-style stds, scaled normal init, embedding norm scaling, or orthogonal init.
- Logit softcap `LOGIT_SOFTCAP=30.0` via `c * tanh(logits / c)`: chosen to bound extreme logits, which helps stability in small models and during quantized export. Common alternatives: raw logits, label smoothing, logit clipping only at eval, or temperature schedules.
- `COMPUTE_DTYPE=bfloat16` in MLX and bf16 forward compute in PyTorch with fp32 master/control storage: chosen because bf16 is the natural speed/quality tradeoff on H100s and Apple Silicon while fp32 storage preserves optimizer quality for sensitive tensors. Common alternatives: fp16, full fp32, fp8, or per-parameter mixed storage policies.
- `CastedLinear` with fp32 weights cast to compute dtype at matmul time: chosen to keep optimizer state and weight updates stable while still running bf16 compute. Common alternatives: native bf16 parameter storage, AMP-only casting, or keeping everything fp32.
- Restore low-dimensional and named control tensors to fp32 in PyTorch: chosen because those tensors are small, cheap to keep precise, and disproportionately important to behavior. Common alternatives: keep everything in bf16, whitelist fewer tensors, or keep only optimizer states in fp32.
- `LOGIT_CHUNK_TOKENS=0` by default in MLX: chosen because the fastest path is a single projection and cross-entropy when memory allows. Common alternatives: chunked logits by default for safety or adaptive chunk sizing by available memory.

## Optimization and schedule choices

- Optimizer split by parameter type: embeddings and scalar/control tensors use Adam, while 2D block matrices use Muon. Chosen because matrix-shaped updates benefit from Muon's orthogonalization, while embeddings and control vectors often prefer a standard adaptive optimizer. Common alternatives: all-AdamW, all-Muon, Shampoo, Adafactor, Lion, or SGD variants.
- `EMBED_LR=0.6` for untied embeddings in PyTorch: likely chosen because untied input embeddings can move aggressively without immediately destabilizing the output head. Common alternatives: a single LR for the whole model or smaller embed LR close to the matrix LR.
- `HEAD_LR=0.008` for an untied output head: likely chosen because the output head is sensitive and can destabilize loss quickly in small models if moved too hard. Common alternatives: same LR as embeddings, same LR as matrices, or no separate head LR.
- `TIED_EMBED_LR=0.05`: chosen because tied embeddings are simultaneously input and output weights, so they usually need a more conservative LR than a standalone embedding matrix. Common alternatives: share the matrix LR, use AdaFactor-style schedule scaling, or freeze embeddings for early steps.
- `MATRIX_LR=0.04`: chosen as the main Muon learning rate for block matrices. Common alternatives: lower Muon rates for stability, higher rates with stronger warmup, or layerwise LR scaling.
- `SCALAR_LR=0.04`: chosen so the cheap control tensors move on roughly the same scale as the main matrices while still using Adam. Common alternatives: a lower scalar LR to reduce overfitting of gates or a higher LR to accelerate residual reweighting.
- `BETA1=0.9`, `BETA2=0.95`, `ADAM_EPS=1e-8`: chosen as a fast-training Adam setting with a shorter second-moment horizon than the GPT default `0.999`. Common alternatives: AdamW defaults like `0.9/0.95` or `0.9/0.999`, Adafactor-style second moments, or epsilon tuning for bf16.
- `MUON_MOMENTUM=0.95`: chosen to give Muon strong temporal smoothing without making the update too sluggish. Common alternatives: lower momentum around `0.9`, higher momentum around `0.98`, or no momentum warmup.
- `MUON_BACKEND_STEPS=5`: chosen as a middle ground between orthogonalization quality and compute overhead. Common alternatives: fewer Newton-Schulz iterations for speed or more iterations for a tighter orthogonal update.
- `MUON_MOMENTUM_WARMUP_START=0.85` and `MUON_MOMENTUM_WARMUP_STEPS=500`: chosen to ramp Muon into its full momentum rather than applying the strongest momentum immediately at initialization. Common alternatives: fixed momentum from step 0 or a longer warmup tied to total steps.
- `GRAD_CLIP_NORM=0.0` by default: chosen because the baseline expects the optimizer split and initialization to be stable enough without clipping, and clipping can hide useful signals. Common alternatives: global clipping, per-parameter clipping, or adaptive clipping.
- `TRAIN_BATCH_TOKENS=524288`: chosen to maximize signal per optimizer step while staying inside the intended hardware envelope. Common alternatives: smaller batches for more frequent updates, larger batches with stronger LR scaling, or curriculum batch sizing.
- `ITERATIONS=20000`: chosen to provide a long default budget for both the 10-minute path and longer unconstrained experiments, with wallclock stopping as the real limiter. Common alternatives: fixed time-only training, fewer steps with larger batches, or dataset-epoch-based stopping.
- `GRAD_ACCUM_STEPS=8 // WORLD_SIZE` in PyTorch: chosen so the effective global batch stays constant as the number of CUDA processes changes, but only for world sizes that divide 8. Common alternatives: fixed per-rank microbatches, any world size plus fractional batch adjustments, or a separate global batch calculation.
- `GRAD_ACCUM_STEPS=8` in MLX: chosen to mirror the single-process effective batch behavior of the PyTorch baseline. Common alternatives: no accumulation on Macs, smaller accumulation for latency, or dynamic accumulation by memory headroom.
- `WARMUP_STEPS=20`: chosen to prime compiled kernels and early optimization dynamics without spending much of the 10-minute budget. Common alternatives: no warmup, a longer LR warmup, or separate compile-only warmup and LR warmup.
- `WARMDOWN_ITERS=1200`: chosen to taper the LR near the end, especially when the wallclock cap is about to hit. Common alternatives: cosine decay over the full run, no decay, step decay, or one-cycle schedules.
- `MAX_WALLCLOCK_SECONDS=600.0`: chosen because the challenge target is a 10-minute training budget on 8xH100s. Common alternatives: unlimited wallclock for non-record research or machine-specific caps.
- Time-aware linear warmdown instead of a pure step-based schedule: chosen so LR decay still behaves sensibly when the wallclock cap truncates the run early. Common alternatives: fixed-step schedules, cosine by iteration, or no schedule tied to elapsed time.
- Periodic validation and log-first-step behavior: chosen so the run remains observable from the beginning without adding too much overhead. Common alternatives: late-start logging, final-only validation, or asynchronous validation.

## Systems choices

- `torch.compile(..., dynamic=False, fullgraph=True)` on the CUDA model: chosen to maximize graph capture and fused execution on a fixed-shape training loop. Common alternatives: eager PyTorch, partial compilation, Triton custom kernels, or handwritten CUDA.
- Compile the Muon backend function separately in PyTorch: chosen because the Newton-Schulz loop is repeated and shape-stable, so compile overhead can amortize well. Common alternatives: eager execution or moving the optimizer math into a custom kernel.
- DDP with `broadcast_buffers=False`: chosen because the model has no meaningful running buffers that need synchronization every step, and the setting saves overhead. Common alternatives: `broadcast_buffers=True`, FSDP, ZeRO, or single-process multi-GPU.
- Require `WORLD_SIZE` to divide 8 in PyTorch: chosen to preserve the intended effective batch shape and keep the baseline simple. Common alternatives: support arbitrary world sizes or infer accumulation from device memory.
- Enable Flash SDP only and disable cuDNN, math, and mem-efficient SDP backends: chosen to force the fast path the authors likely benchmarked. Common alternatives: auto backend choice, mem-efficient attention, or custom attention kernels.
- Enable TF32 on CUDA matmul and cuDNN: chosen to allow fast tensor-core execution for fp32-ish paths without writing extra mixed-precision code. Common alternatives: strict fp32 or fully bf16/fp16-only kernels.
- MLX compile capture over the full model state, not just trainable parameters: chosen because MLX otherwise errors on uncaptured arrays such as RoPE state. Common alternatives: eager MLX execution, more manual state threading, or a different compile boundary.
- `MLX_MAX_MICROBATCH_TOKENS=8192`: chosen to keep peak unified-memory usage low on Macs while preserving the same logical optimizer batch. Common alternatives: a larger chunk for throughput, a smaller emergency chunk, or no intra-microbatch chunking.
- Chunked loss-and-grad accumulation in MLX: chosen because local Apple Silicon runs are memory-limited before they are FLOP-limited. Common alternatives: full-batch gradient evaluation, gradient checkpointing, or lower sequence length.

## Serialization and artifact choices

- Always save a raw model artifact first: chosen because it is useful for debugging and direct reload even though it is not the submission artifact. Common alternatives: quantized-only export or framework-specific checkpoint formats only.
- Always produce an int8 + zlib artifact and validate it with a roundtrip eval: chosen because submission size, not training precision, is the true endpoint metric. Common alternatives: fp16 export, post-training quantization without validation, QAT, or more exotic codecs.
- Per-row int8 quantization for 2D tensors: chosen because rows usually map to output channels and have different scales, so per-row quantization compresses better than one tensor-wide scale. Common alternatives: per-tensor quantization, per-column quantization, blockwise quantization, or 4-bit schemes.
- Per-tensor int8 quantization for vectors and scalars: chosen to keep metadata cheap where per-row structure does not help much. Common alternatives: per-element scaling, keeping all vectors in float, or block quantization.
- `INT8_KEEP_FLOAT_MAX_NUMEL=65536`: chosen because small tensors often do not justify separate quantization metadata and are cheap to preserve. Common alternatives: quantize everything, keep a smaller or larger float passthrough threshold, or threshold by bytes instead of element count.
- `CONTROL_TENSOR_NAME_PATTERNS` and `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`: chosen to preserve numerically sensitive gates, scales, and skip controls in higher precision. Common alternatives: purely size-based rules, hand-written tensor allowlists, or quantizing all tensors equally.
- Store float passthrough tensors in fp16 unless they are explicitly protected: chosen to cut bytes while retaining better fidelity than int8. Common alternatives: fp32 passthrough everywhere, bf16 passthrough, or int8 all the way down.
- `INT8_PER_ROW_SCALE_DTYPE=float16`: chosen to shrink quantization metadata while usually keeping scales accurate enough. Common alternatives: fp32 scales, bf16 scales, or log-scale encodings.
- `INT8_CLIP_PERCENTILE=99.99984`: chosen to clip only extreme outliers before quantization, reducing the damage caused by rare large values. Common alternatives: no clipping, max-abs scaling, lower clip percentiles, or learned clipping.
- zlib level 9 compression: chosen because submission size matters more than serialization speed at the end of training. Common alternatives: lower zlib levels, gzip, zstd, arithmetic coding, or custom entropy coding.
- Load the quantized artifact back into the same model and re-run validation: chosen because the score only matters if the compact artifact still works. Common alternatives: trust the serializer, unit-test only tensor shapes, or validate only a small sample.

## MLX-specific deviations from the CUDA path

- MLX only supports tied embeddings in the starter script: chosen to keep the local baseline small and avoid maintaining a feature branch that is less important for the intended fast iteration loop. Common alternatives: mirror the untied-head option or remove the option from PyTorch too.
- MLX keeps optional logit chunking, while the common path leaves it disabled: chosen because Macs are more likely to run out of memory on the final logits projection. Common alternatives: always chunk logits, reduce batch size instead, or use sampled softmax-style approximations.
- MLX default validation cadence is final-only: chosen because full validation on Apple Silicon can dominate local experiment time. Common alternatives: periodic validation on a smaller subset or a user-configured default cadence above zero.
- MLX logs `tok_s` explicitly per step: chosen because local iteration quality often depends as much on throughput as on final loss. Common alternatives: omit throughput from logs or track hardware counters externally.

## What the baseline is optimizing for

- The baseline is not trying to be the strongest possible model.
- It is trying to be a readable, reproducible starting point that already respects the actual challenge constraints: tiny artifact size, tokenizer-aware evaluation, fast training, and post-training compression.
- The dominant design bias is therefore "cheap control and stability per parameter," not "maximal expressivity regardless of export cost."
