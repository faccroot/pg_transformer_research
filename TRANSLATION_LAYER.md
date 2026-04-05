# Translation Layer

This repo grew bottom-up from experiments, so many of our internal names do not
match the names the literature would use for closely related ideas.

This file is the public translation layer.

The rule is:

- say `closest to`, `bridge into`, or `empirical version of`
- say `inspired by` only when we explicitly built from prior published work
- do not claim we implemented a literature object if we only built something adjacent

The point is not to retrofit buzzwords. It is to make clear that several things
we found empirically are the same family of ideas the field is converging on.

## Short Version

The shortest literature-language summary of this repo is:

- discrete latent state transfer
- chunk-clock recurrent memory
- latent prediction instead of token-only supervision
- low-bit stateful modules
- adaptive chunk-level compute control
- random-basis structural priors

That is why the project maps onto several active research fronts at once even
though the internal names are local to this codebase.

## Core Translation Rule

Good:

- `state_book` is closest to a VQ-style discrete latent codebook object.
- `segment-prev-read` is our chunk-clock memory object.
- the manager/helper bus is a bridge into adaptive compute and TTT-style updates.
- harmonic and boundary work are bridges into H-net-style hierarchical tokenization.

Avoid:

- `we implemented VQ-VAE`
- `we built BLT`
- `we already have a universal transformer`
- `the model trains itself`

Those overstate what is in the repo today.

## Architecture Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| `state_book` | discrete latent state vocabulary, VQ-style codebook object | a small learned structural state basis transferred into the LM |
| hardmax structural controller | discrete latent controller, structural state machine | a low-bandwidth recurrent structural path that shapes the softmax LM |
| hardmax/softmax split | dual-process or neuro-symbolic hybrid | deterministic structural state plus probabilistic token inference |
| sidecar | structural adapter, auxiliary state injection | a lightweight structural module providing always-on conditioning |
| interleaved hardmax/softmax heads | heterogeneous-head attention | mixed discrete/probabilistic mechanisms inside one layer |
| softmax trunk | autoregressive language model trunk | the standard GPT-style LM doing next-token prediction |

## Training Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| execution-trace pretraining | self-generated algorithmic state supervision | pretraining on synthetic VM traces, not yet LM self-tracing |
| trace transfer | execution-supervised latent transfer | transfer the learned structural vocabulary into language modeling |
| `freeze200` / `freeze300` | frozen-prior adaptation, delayed codebook drift | freeze the transferred vocabulary early so the trunk learns to use it before it moves |
| controller collapse | codebook collapse, discrete latent collapse | the LM learns to ignore the discrete state and overuses one or two codes |
| anti-collapse pressure | occupancy objectives, commitment-style regularization | explicit losses to keep the discrete state alive and used |
| supervision density | dense auxiliary supervision | adding more useful training signal per position than next-token CE alone |
| multi-horizon prediction | multi-step auxiliary prediction | predict multiple future horizons instead of only the next token |
| self-distillation with hardmax verification | consistency-filtered self-distillation | use discrete structural agreement as a training-time verifier |

## Memory / Context Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| `segment-prev-read` | chunk-clock recurrent memory, compressed segment memory | read the previous segment summary at a slower clock than token rate |
| `LONGCTX_PERSIST_TRAIN_CARRY=0` | negative result against token-rate recurrent carry | evidence that naive fast recurrent carry is the wrong memory object here |
| sidecar JEPA losses | latent transition or latent prediction objectives | auxiliary prediction in hidden-state space rather than only token space |
| prefix compiler | compiled prefix memory, latent cache compression | compact carryover object distilled from earlier context |
| boundary-collapse bug | boundary-placement / representation entanglement | one latent was trying to both place boundaries and predict summaries |
| detached boundary head | boundary router separated from summary latent | split chunk placement from representation prediction |
| harmonic / chord / patch segmentation | hierarchical chunking, H-net-style token grouping bridge | learned or semi-learned larger units above raw token rate |
| prosody / quote / punctuation features | weak boundary priors, text-native structure features | whitespace, quote, and punctuation cues as cheap structural signals |
| KV-cache compaction by state | state-guided cache management | use structural transitions to predict which memory is still relevant |

## Routing / Compute Allocation Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| confidence-based budget routing | adaptive compute allocation | scale compute by structural confidence; currently negative in our regime |
| attention temperature modulation | attention sharpness modulation | use structural confidence to change attention entropy rather than skip compute |
| always-on conditioning | persistent structural conditioning | continuous injection beat selective routing in our experiments |
| phase-gated curriculum | curriculum learning with phase gating | standard curriculum, but gating the active training regime by phase |

## Quantization / Compression Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| TurboQuant | quantization-aware training pipeline | our QAT-aware pretraining and export path |
| ternary state modules | low-bit stateful subgraph quantization | quantize the new stateful modules first, not the whole trunk first |
| `6.9MB` trusted base | under-cap compressed foundation | a strong artifact using less than half the byte budget, leaving room for structure |
| rotated ternary branch | low-bit geometry-aware compression branch | alternate compression lane that is checkpoint-sensitive rather than default-promoted |

## Representation / Transfer Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| geometry prior | random-basis or stitched structural prior | compressed representational prior from other models |
| ecology model | multi-source routing over pretrained models | a learned router over which source model represents a region best |
| activation merge | representation merging, activation ensembling | combine complementary hidden representations directly |
| random-map adapters | random-basis adapters | closest to the same family as VeRA / RandLoRA-style surfaces |
| random structural baseline | matched-capacity structural prior control | the right comparison for transfer claims, not plain control only |
| scalar forward signatures | behavioral fingerprinting | tested and strongly negative; scalar projections destroyed the geometric signal |

## Systems / Training Infrastructure Concepts

| Our name | Closest literature term | What we actually mean |
| --- | --- | --- |
| TrainOS | adaptive training orchestration, training-time compute allocation | student + manager + helper-worker async system |
| teacher-hidden worker | asynchronous distillation or teacher caching | helper process generating hidden targets outside the student critical path |
| manager process | training controller, runtime adaptation | external process adjusting training parameters from live metrics |
| replay queue | hard-example replay, asynchronous curriculum | emit difficult spans and feed them back later |
| grouped slim trainer | order-of-operations optimization, pre-megakernel systems win | lighter trainer loop and grouped streaming that improve BPB before fusion work |
| helper/manager bus | adaptive compute controller, TTT substrate | side processes that generate signal or proposals for one priority student |

## Theory-Language Mappings

| Our name | Closest literature framing | What we actually mean |
| --- | --- | --- |
| `softmax is inference, hardmax is logic` | dual-process decomposition | deterministic structural operations and probabilistic inference are separated |
| `Farmer decomposition` | mechanical floor plus strategic residual | a large share of prediction is mechanically constrained, the rest is the interesting residual |
| typed tokens / mechanism labels | mechanism-aware tokenization | label events by generating process, not just observed symbol value |
| `softmax as lossy compiler` | neural program execution / learned interpreters | transformer computation behaves like approximate execution over symbolic sequences |

## Key Negative Results, Translated Correctly

| Our finding | Literature-language read |
| --- | --- |
| token-rate carry is dead | chunk-level or segment-level updates dominated token-rate recurrence in our regime |
| budget routing hurts | always-on conditioning outperformed selective compute gating |
| scalar features destroy geometric signal | scalar summaries were too lossy for multi-model routing or structural transfer |
| controller collapses under weak structural labels | known discrete latent collapse failure mode |
| harmonic-local-only was mostly a trainer-loop artifact | apparent architectural gain was confounded with schedule / loop structure |
| larger students regressed | supervision quality dominated raw student capacity in that transfer regime |

## The Two Most Important Translation Claims

1. The hardmax lane is best described as:
   execution-trace-supervised discrete latent transfer into a language model

2. The segment-memory lane is best described as:
   chunk-clock recurrent memory with boundary-routing pressure, where token-rate
   carry was a negative result

Those are the clearest bridges from our own vocabulary to the language the
field already uses.

## One-Paragraph Translations For Common Audiences

### For ML Researchers

We are exploring discrete latent structural controllers for parameter-constrained
language models. The strongest result so far is a VQ-style state vocabulary
pretrained on execution traces from a synthetic VM, then transferred into a GPT
trunk with staged unfreezing, producing a material BPB improvement on FineWeb.
The discrete states still partially collapse inside the LM, so the next step is
anti-collapse objectives and latent state-transition prediction, not opening a
completely different branch.

### For Systems / Infra Readers

We built an async training orchestration layer where a student model trains on
the primary objective while helper workers asynchronously generate replay and
teacher signals, and an external manager process adapts training parameters from
live metrics. The whole system runs on a 14-node Mac Mini cluster with remote
dispatch, artifact recovery, and automated post-run diagnostics.

### For Competition Reviewers

Our strongest trusted baseline reaches `1.43412685` exact BPB at only
`6,925,213` bytes, leaving roughly `9 MB` of headroom under the cap. On top of
that we have signs-of-life in discrete latent transfer, chunk-clock memory,
trainer-loop systems gains, and adaptive compute. The H100 ask is mainly to
stack validated wins and harden the weird branches, not to start from zero.

### For Frontier-Research Conversations

The core thesis is that small models are bottlenecked less by parameter count
than by weak supervision on latent state. We attack that with execution-trace
supervision, chunk-clock memory, multi-horizon prediction, low-bit stateful
modules, and geometry-based transfer priors. The striking empirical result is
that structural state learned from synthetic execution traces transfers into
natural-language compression.
