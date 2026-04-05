# State-Token JEPA For Parameter Golf

`train_state_jepa_mlx.py` is the first pass at a LeWorldModel-style language model in the Parameter Golf codebase.

## Core idea

Instead of an external `encode -> predict -> decode` pipeline, the latent state is part of the sequence itself.

For each chunk of `K` tokens, the mixed sequence is:

`[prior_state_t, x_{t,1}, ..., x_{t,K}, posterior_state_t]`

The transformer processes that augmented sequence causally.

- `prior_state_t` can only use past context.
- token positions inside chunk `t` can condition on `prior_state_t` plus earlier tokens.
- `posterior_state_t` can summarize the observed chunk after seeing its tokens.

## Loss

Training optimizes:

`CE(tokens) + STATE_PRED_WEIGHT * MSE(pred(prior_state_t), posterior_state_t) + SIGREG_WEIGHT * SIGReg(posterior_state_t)`

So:

- token CE is the observation/readout model needed for BPB
- latent prediction is the chunk-level dynamics objective
- SIGReg keeps the state space non-collapsed and close to isotropic

## Why this is different from `train_jepa.py`

`train_jepa.py` is a hierarchical latent LM with separate encoder / predictor / decoder modules.

`train_state_jepa_mlx.py` instead makes the latent state a first-class object inside the backbone:

- no separate chunk encoder or decoder
- cross-chunk token attention is preserved
- the state dynamics live in the same transformer that produces token logits

## Status

- Implemented as an MLX trainer.
- Syntax-checked locally.
- Intended next step is Mini-cluster smoke testing and head-to-head comparison with the baseline and `train_jepa.py`.
