from __future__ import annotations

import math
from pathlib import Path

import numpy as np


ROLE_PARAMETER_SUFFIXES: dict[str, str] = {
    "q": "attn.c_q.weight",
    "k": "attn.c_k.weight",
    "v": "attn.c_v.weight",
    "o": "attn.proj.weight",
    "mlp_fc": "mlp.fc.weight",
    "mlp_proj": "mlp.proj.weight",
}

TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "q": ("q",),
    "k": ("k",),
    "v": ("v",),
    "o": ("o",),
    "proj": ("o",),
    "qk": ("q", "k"),
    "qkv": ("q", "k", "v"),
    "qkvo": ("q", "k", "v", "o"),
    "attn": ("q", "k", "v", "o"),
    "fc": ("mlp_fc",),
    "mlp_in": ("mlp_fc",),
    "mlp_fc": ("mlp_fc",),
    "mlp_out": ("mlp_proj",),
    "mlp_proj": ("mlp_proj",),
    "mlp": ("mlp_fc", "mlp_proj"),
    "all": ("q", "k", "v", "o", "mlp_fc", "mlp_proj"),
}

ROLE_SEED_OFFSETS: dict[str, int] = {
    "q": 17,
    "k": 29,
    "v": 43,
    "o": 59,
    "mlp_fc": 71,
    "mlp_proj": 89,
}


def load_geometry(path: str):
    try:
        from tools.representation_learning.schemas import ModelRepresentation, PlatonicGeometry
    except ModuleNotFoundError as exc:
        if exc.name != "tools":
            raise
        from schemas import ModelRepresentation, PlatonicGeometry  # type: ignore[no-redef]

    payload = np.load(path, allow_pickle=False)
    kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
    if kind == "platonic_geometry":
        return PlatonicGeometry.load(path)
    if kind == "model_representation":
        return ModelRepresentation.load(path)
    raise ValueError(f"Unsupported representation-learning artifact kind {kind!r} in {path}")


def select_layer_geometry(layer_geometries: dict[int, object], relative_depth: float):
    if not layer_geometries:
        raise ValueError("representation-learning artifact contains no layer geometries")
    best_idx = min(
        layer_geometries,
        key=lambda idx: abs(float(layer_geometries[idx].relative_depth) - float(relative_depth)),
    )
    return layer_geometries[best_idx]


def orthonormal_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    q, _r = np.linalg.qr(np.asarray(matrix, dtype=np.float32).T, mode="reduced")
    return np.asarray(q.T, dtype=np.float32)


def parse_prior_targets(targets: str) -> list[str]:
    tokens = [token.strip().lower() for token in str(targets).replace("+", ",").split(",") if token.strip()]
    if not tokens:
        raise ValueError("targets must be non-empty")
    expanded: list[str] = []
    for token in tokens:
        alias = TARGET_ALIASES.get(token)
        if alias is None:
            raise ValueError(f"Unsupported prior target {token!r}")
        for role in alias:
            if role not in expanded:
                expanded.append(role)
    return expanded


def parameter_key_for_role(block_idx: int, role: str) -> str:
    suffix = ROLE_PARAMETER_SUFFIXES.get(role)
    if suffix is None:
        raise KeyError(f"Unknown runtime prior role {role!r}")
    return f"blocks.{block_idx}.{suffix}"


def _random_projection(source_dim: int, target_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((source_dim, target_dim), dtype=np.float32) / max(math.sqrt(source_dim), 1.0)


def _carrier_basis_from_matrix(
    matrix: np.ndarray,
    *,
    basis_dim: int,
    axis: str,
    seed: int,
) -> np.ndarray | None:
    source = np.asarray(matrix, dtype=np.float32)
    target_dim = int(source.shape[1] if axis == "in" else source.shape[0])
    if basis_dim <= 0 or basis_dim > target_dim:
        return None
    u, _s, vh = np.linalg.svd(source, full_matrices=False)
    if axis == "in":
        rows = np.asarray(vh[:basis_dim], dtype=np.float32)
    elif axis == "out":
        rows = np.asarray(u[:, :basis_dim].T, dtype=np.float32)
    else:
        raise ValueError(f"axis must be 'in' or 'out', got {axis!r}")
    if rows.shape[0] < basis_dim:
        extra = _random_projection(basis_dim - rows.shape[0], target_dim, seed=seed)
        rows = np.concatenate([rows, extra], axis=0)
    return np.asarray(rows[:basis_dim], dtype=np.float32)


def map_directions(
    directions: np.ndarray,
    target_dim: int,
    seed: int,
    *,
    carrier_basis: np.ndarray | None = None,
) -> np.ndarray:
    source = np.asarray(directions, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"directions must have shape [k, dim], got {source.shape}")
    if source.shape[0] == 0:
        return np.zeros((0, target_dim), dtype=np.float32)
    if source.shape[1] == target_dim:
        return orthonormal_rows(source)
    projection = None
    if carrier_basis is not None:
        carrier = np.asarray(carrier_basis, dtype=np.float32)
        if carrier.shape == (source.shape[1], target_dim):
            projection = carrier
    if projection is None:
        projection = _random_projection(source.shape[1], target_dim, seed=seed)
    return orthonormal_rows(source @ projection)


def _match_frobenius_norm(target: np.ndarray, current: np.ndarray | None) -> np.ndarray:
    matched = np.asarray(target, dtype=np.float32)
    if current is None:
        return matched
    current_norm = float(np.linalg.norm(np.asarray(current, dtype=np.float32)))
    target_norm = float(np.linalg.norm(matched))
    if current_norm <= 1e-8 or target_norm <= 1e-8:
        return matched
    return matched * (current_norm / target_norm)


def construct_matrix(
    directions: np.ndarray,
    scales: np.ndarray | None,
    *,
    out_dim: int,
    in_dim: int,
    seed: int,
    current_matrix: np.ndarray | None = None,
    adapter_mode: str = "random",
) -> np.ndarray:
    adapter_mode_norm = str(adapter_mode).strip().lower()
    input_carrier = None
    output_carrier = None
    if adapter_mode_norm in {"svd_carrier", "svd_carrier_matched"} and current_matrix is not None:
        input_carrier = _carrier_basis_from_matrix(
            current_matrix,
            basis_dim=int(np.asarray(directions).shape[1]),
            axis="in",
            seed=seed,
        )
        output_carrier = _carrier_basis_from_matrix(
            current_matrix,
            basis_dim=int(np.asarray(directions).shape[1]),
            axis="out",
            seed=seed + 1,
        )
    elif adapter_mode_norm != "random":
        raise ValueError(f"Unsupported adapter_mode {adapter_mode!r}")
    basis_in = map_directions(directions, target_dim=in_dim, seed=seed, carrier_basis=input_carrier)
    basis_out = map_directions(directions, target_dim=out_dim, seed=seed + 1, carrier_basis=output_carrier)
    active = min(basis_in.shape[0], basis_out.shape[0])
    if active <= 0:
        return np.zeros((out_dim, in_dim), dtype=np.float32)
    basis_in = basis_in[:active]
    basis_out = basis_out[:active]
    if scales is None:
        scale_vector = np.ones((active,), dtype=np.float32)
    else:
        scale_vector = np.asarray(scales, dtype=np.float32).reshape(-1)[:active]
        scale_vector = scale_vector / max(float(np.mean(np.abs(scale_vector))), 1e-6)
    target = np.asarray(basis_out.T @ np.diag(scale_vector) @ basis_in, dtype=np.float32)
    if adapter_mode_norm == "svd_carrier_matched":
        target = _match_frobenius_norm(target, current_matrix)
    return target


def summarize_matrix_update(
    current: np.ndarray,
    target: np.ndarray,
    updated: np.ndarray,
    *,
    role: str,
    weight_key: str,
    blend_strength: float,
) -> dict[str, float | int | str]:
    current_norm = float(np.linalg.norm(current))
    target_norm = float(np.linalg.norm(target))
    updated_norm = float(np.linalg.norm(updated))
    delta = updated - current
    delta_norm = float(np.linalg.norm(delta))
    denom = max(current_norm, 1e-6)
    return {
        "role": role,
        "weight_key": weight_key,
        "out_dim": int(current.shape[0]),
        "in_dim": int(current.shape[1]),
        "blend_strength": float(blend_strength),
        "current_norm": current_norm,
        "target_norm": target_norm,
        "updated_norm": updated_norm,
        "delta_norm": delta_norm,
        "delta_ratio_vs_current": delta_norm / denom,
        "target_ratio_vs_current": target_norm / denom,
    }


def apply_priors(
    model,
    *,
    priors_path: str,
    strength: float = 0.5,
    targets: str = "qk",
    adapter_mode: str = "random",
) -> dict[str, object]:
    import mlx.core as mx
    import train_gpt_mlx as base

    if not priors_path:
        raise ValueError("REP_LEARN_QK_INIT=1 requires REP_LEARN_PRIORS_PATH")
    resolved = Path(priors_path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Representation-learning priors not found: {resolved}")
    clipped_strength = float(np.clip(strength, 0.0, 1.0))
    if clipped_strength <= 0.0:
        return {
            "applied_role_count": 0,
            "applied_roles": [],
            "adapter_mode": str(adapter_mode).strip().lower(),
            "layers": [],
            "priors_path": str(resolved),
            "strength": clipped_strength,
        }
    roles = parse_prior_targets(targets)
    geometry = load_geometry(str(resolved))
    flat_state = base.flat_parameter_state(model)
    updated = dict(flat_state)
    block_count = len(model.blocks)
    diagnostics: list[dict[str, object]] = []
    for block_idx in range(block_count):
        relative_depth = (block_idx + 1) / max(block_count, 1)
        layer_geometry = select_layer_geometry(geometry.layer_geometries, relative_depth)
        layer_summary: dict[str, object] = {
            "block_idx": int(block_idx),
            "relative_depth": float(relative_depth),
            "source_layer_idx": int(
                min(
                    geometry.layer_geometries,
                    key=lambda idx: abs(float(geometry.layer_geometries[idx].relative_depth) - float(relative_depth)),
                )
            ),
            "roles": {},
        }
        for role in roles:
            weight_key = parameter_key_for_role(block_idx, role)
            if weight_key not in flat_state:
                continue
            current = np.asarray(flat_state[weight_key].astype(mx.float32), dtype=np.float32)
            seed_offset = ROLE_SEED_OFFSETS[role]
            target = construct_matrix(
                layer_geometry.directions,
                layer_geometry.scales,
                out_dim=current.shape[0],
                in_dim=current.shape[1],
                seed=seed_offset + block_idx * 13,
                current_matrix=current,
                adapter_mode=adapter_mode,
            )
            blended = (1.0 - clipped_strength) * current + clipped_strength * target
            updated[weight_key] = mx.array(blended, dtype=flat_state[weight_key].dtype)
            layer_summary["roles"][role] = summarize_matrix_update(
                current,
                target,
                blended,
                role=role,
                weight_key=weight_key,
                blend_strength=clipped_strength,
            )
        if layer_summary["roles"]:
            diagnostics.append(layer_summary)
    base.apply_flat_arrays(model, updated)
    return {
        "applied_role_count": int(sum(len(layer["roles"]) for layer in diagnostics)),
        "applied_roles": roles,
        "adapter_mode": str(adapter_mode).strip().lower(),
        "layers": diagnostics,
        "priors_path": str(resolved),
        "strength": clipped_strength,
    }


def apply_qk_priors(
    model,
    *,
    priors_path: str,
    strength: float = 0.5,
    targets: str = "qk",
    adapter_mode: str = "random",
) -> dict[str, object]:
    return apply_priors(
        model,
        priors_path=priors_path,
        strength=strength,
        targets=targets,
        adapter_mode=adapter_mode,
    )
