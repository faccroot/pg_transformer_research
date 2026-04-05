#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_compiler_transition_bins import build_transition_examples, classify_transition_bin
from text_prosody_features import BOUNDARY_STRENGTH_TO_ID
from torch_artifact_eval_utils import forward_logits_with_prefix, make_model_from_checkpoint, per_token_nll
from train_gpt import limit_validation_tokens, load_validation_tokens
from train_prefix_compiler_ablation import load_compiler_artifact


def mean_of(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def slot_metrics(
    slots: torch.Tensor,
    transition_bins: list[str],
    structure_labels: list[str],
) -> dict[str, object]:
    # slots: [W, S, D]
    slots = F.normalize(slots.float(), dim=-1)
    num_windows, num_slots, _dim = slots.shape

    same_slot_diff_window: list[float] = []
    diff_slot_same_window: list[float] = []
    same_slot_same_bin: list[float] = []
    same_slot_diff_bin: list[float] = []
    same_slot_same_structure: list[float] = []
    same_slot_diff_structure: list[float] = []

    for w in range(num_windows):
        for s1 in range(num_slots):
            v1 = slots[w, s1]
            for s2 in range(s1 + 1, num_slots):
                diff_slot_same_window.append(float(torch.dot(v1, slots[w, s2]).item()))
            for w2 in range(w + 1, num_windows):
                cos_same_slot = float(torch.dot(v1, slots[w2, s1]).item())
                same_slot_diff_window.append(cos_same_slot)
                if transition_bins[w] == transition_bins[w2]:
                    same_slot_same_bin.append(cos_same_slot)
                else:
                    same_slot_diff_bin.append(cos_same_slot)
                if structure_labels[w] == structure_labels[w2]:
                    same_slot_same_structure.append(cos_same_slot)
                else:
                    same_slot_diff_structure.append(cos_same_slot)

    flat = slots.reshape(num_windows * num_slots, -1)
    slot_ids = [slot_idx for _ in range(num_windows) for slot_idx in range(num_slots)]
    bins_flat = [transition_bins[w] for w in range(num_windows) for _ in range(num_slots)]
    structure_flat = [structure_labels[w] for w in range(num_windows) for _ in range(num_slots)]
    nearest_same_slot = 0
    nearest_same_bin = 0
    nearest_same_structure = 0
    neighbors_count = 0
    sims = flat @ flat.T
    for idx in range(flat.size(0)):
        sims[idx, idx] = -1.0
        nn = int(torch.argmax(sims[idx]).item())
        neighbors_count += 1
        nearest_same_slot += int(slot_ids[idx] == slot_ids[nn])
        nearest_same_bin += int(bins_flat[idx] == bins_flat[nn])
        nearest_same_structure += int(structure_flat[idx] == structure_flat[nn])

    return {
        "num_windows": int(num_windows),
        "num_slots": int(num_slots),
        "same_slot_diff_window_mean_cosine": mean_of(same_slot_diff_window),
        "diff_slot_same_window_mean_cosine": mean_of(diff_slot_same_window),
        "same_slot_same_bin_mean_cosine": mean_of(same_slot_same_bin),
        "same_slot_diff_bin_mean_cosine": mean_of(same_slot_diff_bin),
        "same_slot_same_structure_mean_cosine": mean_of(same_slot_same_structure),
        "same_slot_diff_structure_mean_cosine": mean_of(same_slot_diff_structure),
        "nearest_neighbor_same_slot_rate": float(nearest_same_slot / max(neighbors_count, 1)),
        "nearest_neighbor_same_bin_rate": float(nearest_same_bin / max(neighbors_count, 1)),
        "nearest_neighbor_same_structure_rate": float(nearest_same_structure / max(neighbors_count, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compiler-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--val-files", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--eval-windows", type=int, default=12)
    p.add_argument("--seam-tokens", type=int, default=16)
    p.add_argument("--density-window", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = make_model_from_checkpoint(Path(args.checkpoint), device=device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    compiler, compiler_payload = load_compiler_artifact(Path(args.compiler_path), device=device)
    sp = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path).expanduser()))
    val_tokens = limit_validation_tokens(load_validation_tokens(args.val_files, args.seq_len), args.seq_len, args.eval_windows)
    examples = build_transition_examples(
        model,
        val_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        max_windows=args.eval_windows,
        device=device,
    )

    slot_rows: list[dict[str, object]] = []
    prefix_slots: list[torch.Tensor] = []
    transition_bins: list[str] = []
    structure_labels: list[str] = []
    sentence_id = int(BOUNDARY_STRENGTH_TO_ID["sentence"])

    with torch.no_grad():
        for example in examples:
            none_logits, _ = forward_logits_with_prefix(model, example["x_cur"])
            none_nll = per_token_nll(none_logits, example["y_cur"]).squeeze(0).detach().cpu().to(torch.float64)
            prefix = compiler(example["prev_hidden"], example["prev_nll"]).detach().cpu().squeeze(0)
            transition_bin, transition_metrics = classify_transition_bin(
                sp,
                example["prev_token_ids"],
                example["token_ids_cur"],
                seam_tokens=args.seam_tokens,
                density_window=args.density_window,
                curr_none_nll=none_nll,
            )
            structure_label = "|".join(
                [
                    transition_bin,
                    f"prevb{transition_metrics['prev_tail_boundary_max']}",
                    f"currb{transition_metrics['curr_head_boundary_max']}",
                    f"q{transition_metrics['quote_flip']}",
                    f"sent{int(transition_metrics['curr_head_sentence_frac'] >= 0.125)}",
                ]
            )
            prefix_slots.append(prefix)
            transition_bins.append(transition_bin)
            structure_labels.append(structure_label)
            slot_rows.append(
                {
                    "eval_index": int(example["eval_index"]),
                    "transition_bin": transition_bin,
                    "structure_label": structure_label,
                    "transition_metrics": transition_metrics,
                    "slot_norms": [float(v) for v in prefix.float().norm(dim=-1).tolist()],
                }
            )

    slot_tensor = torch.stack(prefix_slots, dim=0)
    payload = {
        "checkpoint": args.checkpoint,
        "compiler_path": args.compiler_path,
        "compiler_config": compiler_payload.get("config", {}),
        "compiler_extra": compiler_payload.get("extra", {}),
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser()),
        "device": str(device),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "eval_windows": int(args.eval_windows),
        "seam_tokens": int(args.seam_tokens),
        "density_window": int(args.density_window),
        "metrics": slot_metrics(slot_tensor, transition_bins, structure_labels),
        "rows": slot_rows,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
