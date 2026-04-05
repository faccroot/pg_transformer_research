#!/usr/bin/env python3
from __future__ import annotations

import os


os.environ.setdefault("CURRICULUM_ENABLED", "1")

import train_gpt_mlx as base


if __name__ == "__main__":
    base.main()
