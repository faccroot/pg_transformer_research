# JEPA For Parameter Golf

`train_jepa.py` is a standalone JEPA-style language model path for the Parameter Golf repo.

## Core idea

The model keeps exact next-token likelihood, but inserts a latent bottleneck at the chunk level:

1. Split the token sequence into fixed chunks of `CHUNK_SIZE`.
2. Encode each target chunk into a latent vector.
3. Predict the next chunk latent autoregressively from previous chunk latents.
4. Decode the predicted latent back into exact token probabilities for the chunk.

This is not a pure latent-MSE JEPA. It is a hierarchical generative JEPA, because the challenge still scores exact BPB.

## Loss

Training mode optimizes:

`CE(pred_latent -> tokens) + PRED_LOSS_WEIGHT * MSE(pred_latent, target_latent) + SIGREG_WEIGHT * SIGReg(target_latent) + DECODER_TEACHER_WEIGHT * CE(target_latent -> tokens)`

Evaluation mode returns only the exact token cross-entropy, so validation BPB remains comparable with the baseline.

## Mapping To Challenge Constraints

- Uses the existing FineWeb shard loader.
- Uses the existing SentencePiece-based BPB accounting.
- Uses the existing quantized export / roundtrip validation path.
- Stays in a single file at the repo root: [train_jepa.py](/home/zaytor/transformer_research/parameter-golf/train_jepa.py)

## Suggested First Run

```bash
RUN_ID=jepa_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
CHUNK_SIZE=8 \
LATENT_DIM=256 \
ENCODER_HIDDEN_DIM=1024 \
NUM_LAYERS=8 \
DECODER_LAYERS=2 \
COMPILE_MODEL=0 \
torchrun --standalone --nproc_per_node=1 train_jepa.py
```

## Current status

- Implemented and syntax-checked.
- CPU smoke-tested for forward/backward shape correctness.
- Not yet tuned for leaderboard competitiveness.
