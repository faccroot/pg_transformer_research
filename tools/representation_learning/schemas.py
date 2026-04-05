from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, default=_json_default)


def _metadata_from_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError("Serialized metadata must decode to a JSON object")
    return payload


def _json_payload_to_str(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def _list_from_json(raw: Any) -> list[Any]:
    text = _json_payload_to_str(raw)
    if not text:
        return []
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Serialized payload must decode to a JSON array")
    return payload


@dataclass
class LayerGeometry:
    relative_depth: float
    directions: np.ndarray
    scales: np.ndarray | None = None
    covariance: np.ndarray | None = None
    coactivation: np.ndarray | None = None
    importance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.relative_depth = float(self.relative_depth)
        self.directions = np.asarray(self.directions, dtype=np.float32)
        if self.directions.ndim != 2:
            raise ValueError(f"directions must have shape [k, dim], got {self.directions.shape}")
        if self.scales is not None:
            self.scales = np.asarray(self.scales, dtype=np.float32).reshape(-1)
            if self.scales.shape[0] != self.directions.shape[0]:
                raise ValueError("scales length must match number of directions")
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=np.float32)
        if self.coactivation is not None:
            self.coactivation = np.asarray(self.coactivation, dtype=np.float32)
        if self.importance is not None:
            self.importance = np.asarray(self.importance, dtype=np.float32).reshape(-1)


@dataclass
class ModelRepresentation:
    model_id: str
    architecture_family: str
    num_parameters: int
    hidden_dim: int
    num_layers: int
    layer_geometries: dict[int, LayerGeometry]
    chunk_losses: np.ndarray | None = None
    chunk_ids: list[str] | None = None
    chunk_layer_projections: dict[int, np.ndarray] = field(default_factory=dict)
    concept_profiles: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["model_representation"]),
            "model_id": np.array([self.model_id]),
            "architecture_family": np.array([self.architecture_family]),
            "num_parameters": np.array([self.num_parameters], dtype=np.int64),
            "hidden_dim": np.array([self.hidden_dim], dtype=np.int32),
            "num_layers": np.array([self.num_layers], dtype=np.int32),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        layer_indices = np.array(sorted(self.layer_geometries), dtype=np.int32)
        payload["layer_indices"] = layer_indices
        for layer_idx in layer_indices.tolist():
            layer = self.layer_geometries[int(layer_idx)]
            prefix = f"layer_{int(layer_idx)}"
            payload[f"{prefix}_relative_depth"] = np.array([layer.relative_depth], dtype=np.float32)
            payload[f"{prefix}_directions"] = layer.directions.astype(np.float32)
            payload[f"{prefix}_metadata_json"] = np.array([_metadata_to_json(layer.metadata)])
            if layer.scales is not None:
                payload[f"{prefix}_scales"] = layer.scales.astype(np.float32)
            if layer.covariance is not None:
                payload[f"{prefix}_covariance"] = layer.covariance.astype(np.float32)
            if layer.coactivation is not None:
                payload[f"{prefix}_coactivation"] = layer.coactivation.astype(np.float32)
            if layer.importance is not None:
                payload[f"{prefix}_importance"] = layer.importance.astype(np.float32)
        if self.chunk_losses is not None:
            payload["chunk_losses"] = np.asarray(self.chunk_losses, dtype=np.float32).reshape(-1)
        if self.chunk_ids is not None:
            payload["chunk_ids"] = np.asarray(self.chunk_ids, dtype=np.str_)
        for layer_idx in sorted(self.chunk_layer_projections):
            payload[f"layer_{int(layer_idx)}_chunk_projections"] = np.asarray(
                self.chunk_layer_projections[int(layer_idx)],
                dtype=np.float32,
            )
        if self.concept_profiles:
            payload["concept_profiles_json"] = np.array([_metadata_to_json(self.concept_profiles)])
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ModelRepresentation":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "model_representation":
            raise ValueError(f"Expected kind=model_representation, got {kind!r}")
        layer_geometries: dict[int, LayerGeometry] = {}
        for layer_idx in np.asarray(payload["layer_indices"], dtype=np.int32).tolist():
            prefix = f"layer_{int(layer_idx)}"
            layer_geometries[int(layer_idx)] = LayerGeometry(
                relative_depth=float(np.asarray(payload[f"{prefix}_relative_depth"]).reshape(-1)[0]),
                directions=np.asarray(payload[f"{prefix}_directions"], dtype=np.float32),
                scales=np.asarray(payload[f"{prefix}_scales"], dtype=np.float32)
                if f"{prefix}_scales" in payload else None,
                covariance=np.asarray(payload[f"{prefix}_covariance"], dtype=np.float32)
                if f"{prefix}_covariance" in payload else None,
                coactivation=np.asarray(payload[f"{prefix}_coactivation"], dtype=np.float32)
                if f"{prefix}_coactivation" in payload else None,
                importance=np.asarray(payload[f"{prefix}_importance"], dtype=np.float32)
                if f"{prefix}_importance" in payload else None,
                metadata=_metadata_from_json(payload[f"{prefix}_metadata_json"]) if f"{prefix}_metadata_json" in payload else {},
            )
        chunk_ids = None
        if "chunk_ids" in payload:
            chunk_ids = [str(value) for value in np.asarray(payload["chunk_ids"]).tolist()]
        chunk_layer_projections: dict[int, np.ndarray] = {}
        for layer_idx in np.asarray(payload["layer_indices"], dtype=np.int32).tolist():
            key = f"layer_{int(layer_idx)}_chunk_projections"
            if key in payload:
                chunk_layer_projections[int(layer_idx)] = np.asarray(payload[key], dtype=np.float32)
        return cls(
            model_id=str(np.asarray(payload["model_id"]).reshape(-1)[0]),
            architecture_family=str(np.asarray(payload["architecture_family"]).reshape(-1)[0]),
            num_parameters=int(np.asarray(payload["num_parameters"]).reshape(-1)[0]),
            hidden_dim=int(np.asarray(payload["hidden_dim"]).reshape(-1)[0]),
            num_layers=int(np.asarray(payload["num_layers"]).reshape(-1)[0]),
            layer_geometries=layer_geometries,
            chunk_losses=np.asarray(payload["chunk_losses"], dtype=np.float32) if "chunk_losses" in payload else None,
            chunk_ids=chunk_ids,
            chunk_layer_projections=chunk_layer_projections,
            concept_profiles=_metadata_from_json(payload["concept_profiles_json"]) if "concept_profiles_json" in payload else {},
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class ForwardSignatureDataset:
    model_id: str
    chunk_ids: list[str]
    top_k: int
    global_features: dict[str, np.ndarray] = field(default_factory=dict)
    layer_features: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    topk_token_ids: np.ndarray | None = None
    topk_token_probs: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_ids = [str(value) for value in self.chunk_ids]
        self.top_k = int(self.top_k)
        num_chunks = len(self.chunk_ids)
        self.global_features = {
            str(name): np.asarray(values, dtype=np.float32).reshape(-1)
            for name, values in self.global_features.items()
        }
        for name, values in self.global_features.items():
            if values.shape[0] != num_chunks:
                raise ValueError(f"global feature {name!r} has {values.shape[0]} rows, expected {num_chunks}")
        normalized_layer_features: dict[int, dict[str, np.ndarray]] = {}
        for layer_idx, payload in self.layer_features.items():
            normalized_payload: dict[str, np.ndarray] = {}
            for name, values in payload.items():
                array = np.asarray(values, dtype=np.float32).reshape(-1)
                if array.shape[0] != num_chunks:
                    raise ValueError(
                        f"layer feature {name!r} for layer {layer_idx} has {array.shape[0]} rows, expected {num_chunks}"
                    )
                normalized_payload[str(name)] = array
            normalized_layer_features[int(layer_idx)] = normalized_payload
        self.layer_features = normalized_layer_features
        if self.topk_token_ids is not None:
            self.topk_token_ids = np.asarray(self.topk_token_ids, dtype=np.int32)
            if self.topk_token_ids.ndim != 2 or self.topk_token_ids.shape[0] != num_chunks:
                raise ValueError("topk_token_ids must have shape [num_chunks, top_k]")
        if self.topk_token_probs is not None:
            self.topk_token_probs = np.asarray(self.topk_token_probs, dtype=np.float32)
            if self.topk_token_probs.ndim != 2 or self.topk_token_probs.shape[0] != num_chunks:
                raise ValueError("topk_token_probs must have shape [num_chunks, top_k]")
        if self.topk_token_ids is not None and self.topk_token_probs is not None:
            if self.topk_token_ids.shape != self.topk_token_probs.shape:
                raise ValueError("topk_token_ids and topk_token_probs must have the same shape")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        global_feature_names = sorted(self.global_features)
        layer_indices = np.array(sorted(self.layer_features), dtype=np.int32)
        layer_feature_names = sorted({name for payload in self.layer_features.values() for name in payload})
        payload: dict[str, Any] = {
            "kind": np.array(["forward_signature_dataset"]),
            "model_id": np.array([self.model_id]),
            "chunk_ids": np.asarray(self.chunk_ids, dtype=np.str_),
            "top_k": np.array([self.top_k], dtype=np.int32),
            "global_feature_names": np.asarray(global_feature_names, dtype=np.str_),
            "layer_indices": layer_indices,
            "layer_feature_names": np.asarray(layer_feature_names, dtype=np.str_),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        for name in global_feature_names:
            payload[f"global_{name}"] = np.asarray(self.global_features[name], dtype=np.float32)
        for layer_idx in layer_indices.tolist():
            for name in layer_feature_names:
                values = self.layer_features.get(int(layer_idx), {}).get(name)
                if values is not None:
                    payload[f"layer_{int(layer_idx)}_{name}"] = np.asarray(values, dtype=np.float32)
        if self.topk_token_ids is not None:
            payload["topk_token_ids"] = np.asarray(self.topk_token_ids, dtype=np.int32)
        if self.topk_token_probs is not None:
            payload["topk_token_probs"] = np.asarray(self.topk_token_probs, dtype=np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ForwardSignatureDataset":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "forward_signature_dataset":
            raise ValueError(f"Expected kind=forward_signature_dataset, got {kind!r}")
        chunk_ids = [str(value) for value in np.asarray(payload["chunk_ids"]).tolist()] if "chunk_ids" in payload else []
        global_features: dict[str, np.ndarray] = {}
        for name in np.asarray(payload["global_feature_names"]).tolist() if "global_feature_names" in payload else []:
            global_features[str(name)] = np.asarray(payload[f"global_{name}"], dtype=np.float32)
        layer_features: dict[int, dict[str, np.ndarray]] = {}
        layer_feature_names = [str(value) for value in np.asarray(payload["layer_feature_names"]).tolist()] if "layer_feature_names" in payload else []
        for layer_idx in np.asarray(payload["layer_indices"], dtype=np.int32).tolist() if "layer_indices" in payload else []:
            features: dict[str, np.ndarray] = {}
            for name in layer_feature_names:
                key = f"layer_{int(layer_idx)}_{name}"
                if key in payload:
                    features[str(name)] = np.asarray(payload[key], dtype=np.float32)
            layer_features[int(layer_idx)] = features
        return cls(
            model_id=str(np.asarray(payload["model_id"]).reshape(-1)[0]),
            chunk_ids=chunk_ids,
            top_k=int(np.asarray(payload["top_k"]).reshape(-1)[0]) if "top_k" in payload else 0,
            global_features=global_features,
            layer_features=layer_features,
            topk_token_ids=np.asarray(payload["topk_token_ids"], dtype=np.int32) if "topk_token_ids" in payload else None,
            topk_token_probs=np.asarray(payload["topk_token_probs"], dtype=np.float32) if "topk_token_probs" in payload else None,
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class PlatonicGeometry:
    canonical_dim: int
    layer_geometries: dict[int, LayerGeometry]
    source_models: list[str] = field(default_factory=list)
    frontier_floor: np.ndarray | None = None
    concept_profiles: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["platonic_geometry"]),
            "canonical_dim": np.array([self.canonical_dim], dtype=np.int32),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
            "layer_indices": np.array(sorted(self.layer_geometries), dtype=np.int32),
        }
        for layer_idx in sorted(self.layer_geometries):
            layer = self.layer_geometries[layer_idx]
            prefix = f"layer_{int(layer_idx)}"
            payload[f"{prefix}_relative_depth"] = np.array([layer.relative_depth], dtype=np.float32)
            payload[f"{prefix}_directions"] = layer.directions.astype(np.float32)
            payload[f"{prefix}_metadata_json"] = np.array([_metadata_to_json(layer.metadata)])
            if layer.scales is not None:
                payload[f"{prefix}_scales"] = layer.scales.astype(np.float32)
            if layer.covariance is not None:
                payload[f"{prefix}_covariance"] = layer.covariance.astype(np.float32)
            if layer.coactivation is not None:
                payload[f"{prefix}_coactivation"] = layer.coactivation.astype(np.float32)
            if layer.importance is not None:
                payload[f"{prefix}_importance"] = layer.importance.astype(np.float32)
        if self.frontier_floor is not None:
            payload["frontier_floor"] = np.asarray(self.frontier_floor, dtype=np.float32).reshape(-1)
        if self.concept_profiles:
            payload["concept_profiles_json"] = np.array([_metadata_to_json(self.concept_profiles)])
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "PlatonicGeometry":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "platonic_geometry":
            raise ValueError(f"Expected kind=platonic_geometry, got {kind!r}")
        layer_geometries: dict[int, LayerGeometry] = {}
        for layer_idx in np.asarray(payload["layer_indices"], dtype=np.int32).tolist():
            prefix = f"layer_{int(layer_idx)}"
            layer_geometries[int(layer_idx)] = LayerGeometry(
                relative_depth=float(np.asarray(payload[f"{prefix}_relative_depth"]).reshape(-1)[0]),
                directions=np.asarray(payload[f"{prefix}_directions"], dtype=np.float32),
                scales=np.asarray(payload[f"{prefix}_scales"], dtype=np.float32)
                if f"{prefix}_scales" in payload else None,
                covariance=np.asarray(payload[f"{prefix}_covariance"], dtype=np.float32)
                if f"{prefix}_covariance" in payload else None,
                coactivation=np.asarray(payload[f"{prefix}_coactivation"], dtype=np.float32)
                if f"{prefix}_coactivation" in payload else None,
                importance=np.asarray(payload[f"{prefix}_importance"], dtype=np.float32)
                if f"{prefix}_importance" in payload else None,
                metadata=_metadata_from_json(payload[f"{prefix}_metadata_json"]) if f"{prefix}_metadata_json" in payload else {},
            )
        source_models = []
        if "source_models" in payload:
            source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()]
        return cls(
            canonical_dim=int(np.asarray(payload["canonical_dim"]).reshape(-1)[0]),
            layer_geometries=layer_geometries,
            source_models=source_models,
            frontier_floor=np.asarray(payload["frontier_floor"], dtype=np.float32) if "frontier_floor" in payload else None,
            concept_profiles=_metadata_from_json(payload["concept_profiles_json"]) if "concept_profiles_json" in payload else {},
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class SharedLatentLayer:
    relative_depth: float
    chunk_ids: list[str]
    shared_latents: np.ndarray
    model_projections: dict[str, np.ndarray]
    model_means: dict[str, np.ndarray]
    aligned_latents: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.relative_depth = float(self.relative_depth)
        self.chunk_ids = [str(value) for value in self.chunk_ids]
        self.shared_latents = np.asarray(self.shared_latents, dtype=np.float32)
        if self.shared_latents.ndim != 2:
            raise ValueError(f"shared_latents must have shape [num_chunks, latent_dim], got {self.shared_latents.shape}")
        num_chunks = len(self.chunk_ids)
        if self.shared_latents.shape[0] != num_chunks:
            raise ValueError("shared_latents row count must match chunk_ids length")
        self.model_projections = {
            str(model_id): np.asarray(matrix, dtype=np.float32)
            for model_id, matrix in self.model_projections.items()
        }
        self.model_means = {
            str(model_id): np.asarray(vector, dtype=np.float32).reshape(-1)
            for model_id, vector in self.model_means.items()
        }
        self.aligned_latents = {
            str(model_id): np.asarray(matrix, dtype=np.float32)
            for model_id, matrix in self.aligned_latents.items()
        }
        latent_dim = int(self.shared_latents.shape[1])
        for model_id, matrix in self.model_projections.items():
            if matrix.ndim != 2 or matrix.shape[1] != latent_dim:
                raise ValueError(f"projection for {model_id!r} must have shape [input_dim, {latent_dim}]")
            mean = self.model_means.get(model_id)
            if mean is None or mean.shape[0] != matrix.shape[0]:
                raise ValueError(f"mean vector for {model_id!r} must have length {matrix.shape[0]}")
            aligned = self.aligned_latents.get(model_id)
            if aligned is not None:
                if aligned.ndim != 2 or aligned.shape != self.shared_latents.shape:
                    raise ValueError(
                        f"aligned_latents for {model_id!r} must have shape {self.shared_latents.shape}, got {aligned.shape}"
                    )


@dataclass
class SharedLatentGeometry:
    latent_dim: int
    input_dim: int
    source_models: list[str]
    layers: dict[int, SharedLatentLayer]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layer_indices = np.array(sorted(self.layers), dtype=np.int32)
        payload: dict[str, Any] = {
            "kind": np.array(["shared_latent_geometry"]),
            "latent_dim": np.array([self.latent_dim], dtype=np.int32),
            "input_dim": np.array([self.input_dim], dtype=np.int32),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "layer_indices": layer_indices,
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        for layer_idx in layer_indices.tolist():
            layer = self.layers[int(layer_idx)]
            prefix = f"layer_{int(layer_idx)}"
            model_ids = sorted(layer.model_projections)
            aligned_model_ids = sorted(layer.aligned_latents)
            payload[f"{prefix}_relative_depth"] = np.array([layer.relative_depth], dtype=np.float32)
            payload[f"{prefix}_chunk_ids"] = np.asarray(layer.chunk_ids, dtype=np.str_)
            payload[f"{prefix}_shared_latents"] = np.asarray(layer.shared_latents, dtype=np.float32)
            payload[f"{prefix}_model_ids"] = np.asarray(model_ids, dtype=np.str_)
            payload[f"{prefix}_aligned_model_ids"] = np.asarray(aligned_model_ids, dtype=np.str_)
            payload[f"{prefix}_metadata_json"] = np.array([_metadata_to_json(layer.metadata)])
            for model_id in model_ids:
                safe_model = model_id.replace("/", "__")
                payload[f"{prefix}_{safe_model}_projection"] = np.asarray(layer.model_projections[model_id], dtype=np.float32)
                payload[f"{prefix}_{safe_model}_mean"] = np.asarray(layer.model_means[model_id], dtype=np.float32)
            for model_id in aligned_model_ids:
                safe_model = model_id.replace("/", "__")
                payload[f"{prefix}_{safe_model}_aligned_latents"] = np.asarray(layer.aligned_latents[model_id], dtype=np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "SharedLatentGeometry":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "shared_latent_geometry":
            raise ValueError(f"Expected kind=shared_latent_geometry, got {kind!r}")
        layers: dict[int, SharedLatentLayer] = {}
        for layer_idx in np.asarray(payload["layer_indices"], dtype=np.int32).tolist():
            prefix = f"layer_{int(layer_idx)}"
            model_ids = [str(value) for value in np.asarray(payload[f"{prefix}_model_ids"]).tolist()]
            aligned_model_ids = [str(value) for value in np.asarray(payload[f"{prefix}_aligned_model_ids"]).tolist()]
            model_projections = {}
            model_means = {}
            aligned_latents = {}
            for model_id in model_ids:
                safe_model = model_id.replace("/", "__")
                model_projections[model_id] = np.asarray(payload[f"{prefix}_{safe_model}_projection"], dtype=np.float32)
                model_means[model_id] = np.asarray(payload[f"{prefix}_{safe_model}_mean"], dtype=np.float32)
            for model_id in aligned_model_ids:
                safe_model = model_id.replace("/", "__")
                aligned_latents[model_id] = np.asarray(payload[f"{prefix}_{safe_model}_aligned_latents"], dtype=np.float32)
            layers[int(layer_idx)] = SharedLatentLayer(
                relative_depth=float(np.asarray(payload[f"{prefix}_relative_depth"]).reshape(-1)[0]),
                chunk_ids=[str(value) for value in np.asarray(payload[f"{prefix}_chunk_ids"]).tolist()],
                shared_latents=np.asarray(payload[f"{prefix}_shared_latents"], dtype=np.float32),
                model_projections=model_projections,
                model_means=model_means,
                aligned_latents=aligned_latents,
                metadata=_metadata_from_json(payload[f"{prefix}_metadata_json"]) if f"{prefix}_metadata_json" in payload else {},
            )
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            latent_dim=int(np.asarray(payload["latent_dim"]).reshape(-1)[0]),
            input_dim=int(np.asarray(payload["input_dim"]).reshape(-1)[0]),
            source_models=source_models,
            layers=layers,
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class DisagreementProbeSet:
    source_models: list[str]
    probes: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["disagreement_probe_set"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "probes_json": np.array([json.dumps(self.probes, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "DisagreementProbeSet":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "disagreement_probe_set":
            raise ValueError(f"Expected kind=disagreement_probe_set, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            source_models=source_models,
            probes=_list_from_json(payload["probes_json"]) if "probes_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class VerificationTable:
    source_models: list[str]
    entries: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["verification_table"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "entries_json": np.array([json.dumps(self.entries, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "VerificationTable":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "verification_table":
            raise ValueError(f"Expected kind=verification_table, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            source_models=source_models,
            entries=_list_from_json(payload["entries_json"]) if "entries_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class RoutingKernel:
    source_models: list[str]
    rules: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["routing_kernel"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "rules_json": np.array([json.dumps(self.rules, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "RoutingKernel":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "routing_kernel":
            raise ValueError(f"Expected kind=routing_kernel, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            source_models=source_models,
            rules=_list_from_json(payload["rules_json"]) if "rules_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class ActivationEventDataset:
    source_models: list[str]
    feature_names: list[str]
    embedding_dim: int
    events: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["activation_event_dataset"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "embedding_dim": np.array([self.embedding_dim], dtype=np.int32),
            "events_json": np.array([json.dumps(self.events, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationEventDataset":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "activation_event_dataset":
            raise ValueError(f"Expected kind=activation_event_dataset, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        feature_names = [str(value) for value in np.asarray(payload["feature_names"]).tolist()] if "feature_names" in payload else []
        return cls(
            source_models=source_models,
            feature_names=feature_names,
            embedding_dim=int(np.asarray(payload["embedding_dim"]).reshape(-1)[0]) if "embedding_dim" in payload else 0,
            events=_list_from_json(payload["events_json"]) if "events_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class EcologyTrainingSet:
    source_models: list[str]
    feature_names: list[str]
    embedding_dim: int
    examples: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["ecology_training_set"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "embedding_dim": np.array([self.embedding_dim], dtype=np.int32),
            "examples_json": np.array([json.dumps(self.examples, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "EcologyTrainingSet":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "ecology_training_set":
            raise ValueError(f"Expected kind=ecology_training_set, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        feature_names = [str(value) for value in np.asarray(payload["feature_names"]).tolist()] if "feature_names" in payload else []
        return cls(
            source_models=source_models,
            feature_names=feature_names,
            embedding_dim=int(np.asarray(payload["embedding_dim"]).reshape(-1)[0]) if "embedding_dim" in payload else 0,
            examples=_list_from_json(payload["examples_json"]) if "examples_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class KernelTeacherDataset:
    source_models: list[str]
    embedding_dim: int
    examples: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["kernel_teacher_dataset"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "embedding_dim": np.array([self.embedding_dim], dtype=np.int32),
            "examples_json": np.array([json.dumps(self.examples, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "KernelTeacherDataset":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "kernel_teacher_dataset":
            raise ValueError(f"Expected kind=kernel_teacher_dataset, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            source_models=source_models,
            embedding_dim=int(np.asarray(payload["embedding_dim"]).reshape(-1)[0]) if "embedding_dim" in payload else 0,
            examples=_list_from_json(payload["examples_json"]) if "examples_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )


@dataclass
class KernelTeacherTextDataset:
    source_models: list[str]
    embedding_dim: int
    examples: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": np.array(["kernel_teacher_text_dataset"]),
            "source_models": np.asarray(self.source_models, dtype=np.str_),
            "embedding_dim": np.array([self.embedding_dim], dtype=np.int32),
            "examples_json": np.array([json.dumps(self.examples, sort_keys=True, default=_json_default)]),
            "metadata_json": np.array([_metadata_to_json(self.metadata)]),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "KernelTeacherTextDataset":
        payload = np.load(Path(path), allow_pickle=False)
        kind = str(np.asarray(payload["kind"]).reshape(-1)[0])
        if kind != "kernel_teacher_text_dataset":
            raise ValueError(f"Expected kind=kernel_teacher_text_dataset, got {kind!r}")
        source_models = [str(value) for value in np.asarray(payload["source_models"]).tolist()] if "source_models" in payload else []
        return cls(
            source_models=source_models,
            embedding_dim=int(np.asarray(payload["embedding_dim"]).reshape(-1)[0]) if "embedding_dim" in payload else 0,
            examples=_list_from_json(payload["examples_json"]) if "examples_json" in payload else [],
            metadata=_metadata_from_json(payload["metadata_json"]),
        )
