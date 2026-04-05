from __future__ import annotations

import json
import platform

import torch

import train_jepa as mod


def main() -> None:
    torch.manual_seed(1234)
    model = mod.JEPALM(
        vocab_size=64,
        chunk_size=4,
        num_layers=2,
        num_layer_templates=2,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
        latent_dim=16,
        encoder_hidden_dim=64,
        decoder_layers=1,
        pred_loss_weight=0.25,
        sigreg_weight=0.05,
        sigreg_knots=9,
        sigreg_num_proj=16,
        decoder_teacher_weight=0.1,
    )
    input_ids = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    target_ids = torch.randint(0, 64, (2, 16), dtype=torch.int64)

    model.train()
    train_loss = model(input_ids, target_ids)
    if not torch.isfinite(train_loss):
        raise RuntimeError(f"Non-finite train loss: {train_loss}")
    train_loss.backward()

    model.eval()
    with torch.no_grad():
        eval_ce = model(input_ids, target_ids)
    if not torch.isfinite(eval_ce):
        raise RuntimeError(f"Non-finite eval CE: {eval_ce}")

    payload = {
        "host": platform.node(),
        "train_loss": float(train_loss.detach().item()),
        "eval_ce": float(eval_ce.detach().item()),
        "breakdown": model.last_loss_breakdown,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
