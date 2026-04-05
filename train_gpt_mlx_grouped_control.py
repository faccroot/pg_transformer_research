#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Callable

import numpy as np

import train_gpt_mlx as base


class Hyperparameters(base.Hyperparameters):
    longctx_num_streams: int = int(os.environ.get("LONGCTX_NUM_STREAMS", "0"))


class GroupedStreamingTokenLoader:
    """Persistent stream lanes for contiguous local-window training."""

    def __init__(
        self,
        pattern: str,
        *,
        num_streams: int,
        seq_len: int,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        if num_streams <= 0:
            raise ValueError(f"LONGCTX_NUM_STREAMS must be > 0, got {num_streams}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.stream = base.TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)
        self.num_streams = int(num_streams)
        self.seq_len = int(seq_len)
        self.windows = [self.stream.take(self.seq_len + 1) for _ in range(self.num_streams)]

    def summary(self) -> dict[str, object]:
        return {
            "mode": "grouped_streaming",
            "num_streams": int(self.num_streams),
            "seq_len": int(self.seq_len),
        }

    def next_batch_np(self, batch_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        batch_seqs = usable // seq_len
        if seq_len != self.seq_len:
            raise ValueError(f"loader seq_len={self.seq_len} does not match request seq_len={seq_len}")
        if batch_seqs != self.num_streams:
            raise ValueError(
                f"grouped streaming expects batch_seqs == num_streams, got batch_seqs={batch_seqs} "
                f"num_streams={self.num_streams}"
            )
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for lane_idx in range(self.num_streams):
            window = self.windows[lane_idx]
            x_rows.append(window[:-1])
            y_rows.append(window[1:])
            next_tokens = self.stream.take(self.seq_len)
            self.windows[lane_idx] = np.concatenate((window[-1:], next_tokens), axis=0)
        x = np.stack(x_rows, axis=0).astype(np.int32, copy=False)
        y = np.stack(y_rows, axis=0).astype(np.int32, copy=False)
        return x, y


_orig_build_train_loader = base.build_train_loader
_orig_gpt_kwargs_from_args = base.gpt_kwargs_from_args


def grouped_build_train_loader(
    args: Hyperparameters,
    log_fn: Callable[[str], None] | None = None,
    dataset_name: str = "",
):
    if args.curriculum_enabled:
        return _orig_build_train_loader(args, log_fn=log_fn, dataset_name=dataset_name)
    batch_seqs = args.microbatch_tokens // args.train_seq_len
    num_streams = args.longctx_num_streams if args.longctx_num_streams > 0 else batch_seqs
    if num_streams != batch_seqs:
        raise ValueError(
            f"LONGCTX_NUM_STREAMS must match microbatch batch size, got {num_streams} vs {batch_seqs}"
        )
    return GroupedStreamingTokenLoader(
        args.train_files,
        num_streams=num_streams,
        seq_len=args.train_seq_len,
        log_fn=log_fn,
        dataset_name=dataset_name,
    )


def grouped_gpt_kwargs_from_args(args: Hyperparameters, sp=None) -> dict[str, object]:
    kwargs = dict(_orig_gpt_kwargs_from_args(args, sp))
    kwargs.pop("prosody_extended_feature_set_enabled", None)
    return kwargs


def main() -> None:
    base.Hyperparameters = Hyperparameters
    base.build_train_loader = grouped_build_train_loader
    base.gpt_kwargs_from_args = grouped_gpt_kwargs_from_args
    base.main()


if __name__ == "__main__":
    main()
