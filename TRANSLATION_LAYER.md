# Translation Layer

This repo grew bottom-up from experiments, so many of our internal names do not
match the names the literature would use for closely related ideas.

This file is the translation layer.

The rule is:

- use `is closest to`, `bridge into`, or `empirical version of`
- avoid claiming we fully implemented a literature object when we did not

## Internal Names -> Literature Terms

| Repo term | Closest literature term | What we actually mean |
| --- | --- | --- |
| `state_book` | discrete latent state vocabulary, VQ-style codebook object | a small learned structural state basis transferred into the LM |
| `freeze200` / `freeze300` | frozen-prior adaptation, delayed codebook drift | freeze the transferred state vocabulary early so the trunk learns to use it before it moves |
| hardmax structural controller | discrete latent controller, compiler-style latent pass | a low-bandwidth structural side path that shapes the softmax LM |
| execution-trace pretraining | self-generated algorithmic state supervision | pretraining on synthetic VM traces, not yet LM self-tracing |
| trace transfer | execution-supervised latent transfer | transfer the learned structural vocabulary into language modeling |
| `segment-prev-read` | chunk-clock recurrent memory, compressed segment memory | read the previous segment summary at a slower clock than token rate |
| `LONGCTX_PERSIST_TRAIN_CARRY=0` | negative result against token-rate recurrent carry | evidence that naive fast recurrent carry is the wrong memory object here |
| sidecar JEPA losses | latent transition / latent prediction objectives | auxiliary prediction in hidden-state space rather than only token space |
| boundary-collapse bug | boundary-placement / representation entanglement | one latent was trying to both place boundaries and predict summaries |
| detached boundary head | boundary router separated from summary latent | split chunk placement from representation prediction |
| harmonic / chord / patch segmentation | hierarchical chunking, H-net-style token grouping bridge | learned or semi-learned larger units above raw token rate |
| prosody / quote / punctuation features | weak boundary priors, text-native structure features | whitespace, quote, and punctuation cues as cheap structural signals |
| SIGReg | latent geometry regularizer, occupancy-spreading regularizer | auxiliary pressure that spreads or stabilizes structural states |
| prefix compiler | compiled prefix memory, latent cache compression | compact carryover object distilled from earlier context |
| grouped slim trainer | order-of-operations optimization, pre-megakernel systems win | lighter trainer loop and grouped streaming that improve BPB before kernel work |
| helper worker / manager bus | adaptive compute controller, TTT substrate | side processes that generate signal or proposals for one priority student |
| replay queue | hard-example replay, asynchronous curriculum | emit difficult spans and feed them back later |
| hidden-state teacher cache | hidden-state distillation substrate | async helper lane writing teacher targets outside the core student step |
| geometry prior | random-basis / stitched structural prior | compressed representational prior from other models |
| random structural baseline | matched-capacity structural prior control | the right comparison for transfer claims, not plain control only |
| ternary state path | low-bit stateful subgraph | quantize the new stateful modules first, not the full trunk first |
| slim-loop / grouped-loop win | systems-side BPB improvement | a training-loop/layout gain that should likely precede fusion work |

## Safe Public Phrasing

Good:

- `state_book` is closest to a VQ-style discrete latent codebook object.
- `segment-prev-read` is our chunk-clock memory object.
- the manager/helper bus is a bridge into adaptive compute and TTT-style updates.
- the harmonic and boundary work is a bridge into H-net-style hierarchical tokenization.

Avoid:

- `we implemented VQ-VAE`
- `we built BLT`
- `we already have a universal transformer`
- `the model trains itself`

Those overstate what is in the repo today.

## Project Summary In Literature Terms

The shortest translation of this repo is:

- discrete latent state transfer
- chunk-clock recurrent memory
- latent prediction rather than token-only supervision
- low-bit stateful modules
- adaptive chunk-level compute control
- random-basis structural priors

That is why the work lines up with several active areas at once even though the
internal names are local to this codebase.

## The Two Most Important Translation Claims

1. The hardmax lane is best described as:
   execution-trace-supervised discrete latent transfer into a language model

2. The segment-memory lane is best described as:
   chunk-clock recurrent memory with boundary-routing pressure, where token-rate
   carry was a negative result

Those two claims are the clearest bridge from our own vocabulary to the
language the field already uses.
