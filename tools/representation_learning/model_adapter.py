from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GradientBatchStatistics:
    importance_sum: np.ndarray
    covariance_numerator: np.ndarray
    weight_denom: float
    loss_sum: float
    token_count: int
    sequence_losses: np.ndarray


@dataclass
class ContrastiveBatchStatistics:
    js_divergence: np.ndarray
    layer_deltas: dict[int, np.ndarray]


@dataclass
class ForwardSignatureBatch:
    global_features: dict[str, np.ndarray]
    layer_features: dict[int, dict[str, np.ndarray]]
    topk_token_ids: np.ndarray
    topk_token_probs: np.ndarray


@dataclass
class SequenceRepresentationBatch:
    mean_hidden: np.ndarray
    last_hidden: np.ndarray
    last_logits: np.ndarray
    attention_mask: np.ndarray | None = None
    layer_last_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    layer_hidden_sequences: dict[int, np.ndarray] = field(default_factory=dict)


class HFCausalLMAdapter:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for the representation-learning extraction stack. "
                "Install requirements-representation-learning.txt on the extraction host."
            ) from exc

        self._torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        dtype: Any = "auto"
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)
        self.tokenizer = self._load_tokenizer(
            AutoTokenizer,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        self.model = self._load_model(
            AutoModelForCausalLM,
            model_id,
            model_kwargs=model_kwargs,
        )
        config = getattr(self.model, "config", None)
        if config is not None:
            if hasattr(config, "attn_implementation"):
                setattr(config, "attn_implementation", "eager")
            if hasattr(config, "_attn_implementation"):
                setattr(config, "_attn_implementation", "eager")
        self.model.to(device)
        self.model.eval()
        self.model_id = model_id
        self.architecture_family = str(getattr(self.model.config, "model_type", "unknown"))
        self.num_layers = int(getattr(self.model.config, "num_hidden_layers", getattr(self.model.config, "n_layer", 0)))
        self.hidden_dim = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 0)))
        self.num_parameters = int(sum(parameter.numel() for parameter in self.model.parameters()))

    def _load_tokenizer(self, auto_tokenizer, model_id: str, *, trust_remote_code: bool):
        try:
            return auto_tokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except ImportError:
            if not trust_remote_code:
                raise
            return auto_tokenizer.from_pretrained(model_id, trust_remote_code=False)

    def _load_model(self, auto_model, model_id: str, *, model_kwargs: dict[str, Any]):
        trust_remote_code = bool(model_kwargs.get("trust_remote_code", False))
        eager_kwargs = dict(model_kwargs)
        eager_kwargs["attn_implementation"] = "eager"
        try:
            return auto_model.from_pretrained(model_id, **eager_kwargs)
        except TypeError:
            pass
        except ImportError:
            if trust_remote_code:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["trust_remote_code"] = False
                try:
                    return auto_model.from_pretrained(model_id, **eager_kwargs | {"trust_remote_code": False})
                except TypeError:
                    return auto_model.from_pretrained(model_id, **fallback_kwargs)
            raise

        try:
            return auto_model.from_pretrained(model_id, **model_kwargs)
        except ImportError:
            if trust_remote_code:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["trust_remote_code"] = False
                try:
                    return auto_model.from_pretrained(model_id, attn_implementation="eager", **fallback_kwargs)
                except TypeError:
                    return auto_model.from_pretrained(model_id, **fallback_kwargs)
            raise

    def tokenize(self, texts: list[str], *, max_length: int | None = None) -> dict[str, Any]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def get_hidden_states(self, texts: list[str], *, max_length: int | None = None) -> dict[int, Any]:
        with self._torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        return {idx: state.detach() for idx, state in enumerate(outputs.hidden_states)}

    def get_mean_pooled_hidden_states(
        self,
        texts: list[str],
        *,
        layers: list[int],
        max_length: int | None = None,
    ) -> dict[int, np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            pooled: dict[int, np.ndarray] = {}
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
                    raise ValueError(f"layer_idx must be in [0, {len(outputs.hidden_states) - 1}], got {layer_idx}")
                pooled[layer_idx] = self._mean_pool_hidden(
                    outputs.hidden_states[layer_idx],
                    encoded.get("attention_mask"),
                ).detach().cpu().numpy().astype(np.float32)
        return pooled

    def get_logit_distribution(self, texts: list[str], *, max_length: int | None = None) -> Any:
        with self._torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(**encoded, use_cache=False)
        return outputs.logits.detach()

    def get_sequence_representations(
        self,
        texts: list[str],
        *,
        layers: list[int] | None = None,
        capture_full_sequences: bool = False,
        max_length: int | None = None,
    ) -> SequenceRepresentationBatch:
        torch = self._torch
        requested_layers = sorted({int(layer_idx) for layer_idx in (layers or [])})
        with torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            final_hidden = outputs.hidden_states[-1]
            attention_mask = encoded.get("attention_mask")
            mean_hidden = self._mean_pool_hidden(final_hidden, attention_mask)
            last_hidden = self._gather_last_token_hidden(final_hidden, attention_mask)
            last_logits = self._gather_last_token_logits(outputs.logits, attention_mask)
            layer_last_hidden: dict[int, np.ndarray] = {}
            layer_hidden_sequences: dict[int, np.ndarray] = {}
            for layer_idx in requested_layers:
                if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
                    raise ValueError(f"layer_idx must be in [0, {len(outputs.hidden_states) - 1}], got {layer_idx}")
                layer_hidden = outputs.hidden_states[layer_idx]
                layer_last_hidden[int(layer_idx)] = self._gather_last_token_hidden(
                    layer_hidden,
                    attention_mask,
                ).detach().cpu().numpy().astype(np.float32)
                if capture_full_sequences:
                    layer_hidden_sequences[int(layer_idx)] = (
                        layer_hidden.detach().cpu().numpy().astype(np.float32)
                    )
        return SequenceRepresentationBatch(
            mean_hidden=mean_hidden.detach().cpu().numpy().astype(np.float32),
            last_hidden=last_hidden.detach().cpu().numpy().astype(np.float32),
            last_logits=last_logits.detach().cpu().numpy().astype(np.float32),
            attention_mask=attention_mask.detach().cpu().numpy().astype(np.int32) if attention_mask is not None else None,
            layer_last_hidden=layer_last_hidden,
            layer_hidden_sequences=layer_hidden_sequences,
        )

    def project_hidden_to_logits(self, hidden: Any) -> Any:
        torch = self._torch
        head = self.model.get_output_embeddings()
        if head is None:
            raise RuntimeError("Model does not expose output embeddings for logit projection")
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(hidden, device=self.device)
        hidden = hidden.to(device=self.device)
        head_dtype = next(head.parameters(), None)
        if head_dtype is None:
            hidden_for_head = hidden
        else:
            hidden_for_head = hidden.to(dtype=head_dtype.dtype)
        logits = head(hidden_for_head)
        return logits.to(dtype=torch.float32)

    def continue_from_layer_hidden_sequence(
        self,
        hidden_sequence: Any,
        attention_mask: Any | None,
        *,
        layer_idx: int,
    ) -> SequenceRepresentationBatch:
        torch = self._torch
        if not hasattr(self.model, "model"):
            raise RuntimeError("Layer continuation requires a decoder-style model with a `.model` module")
        backbone = self.model.model
        if not hasattr(backbone, "layers") or not hasattr(backbone, "norm") or not hasattr(backbone, "rotary_emb"):
            raise RuntimeError("Layer continuation requires `.model.layers`, `.model.norm`, and `.model.rotary_emb`")

        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx > int(self.num_layers):
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}], got {layer_idx}")

        hidden_states = self._to_torch_tensor(hidden_sequence, dtype=torch.float32)
        attention_mask_tensor = None
        if attention_mask is not None:
            attention_mask_tensor = self._to_torch_tensor(attention_mask, dtype=torch.long)
        position_ids = self._default_position_ids(hidden_states, attention_mask_tensor)
        causal_masks = self._build_causal_masks(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask_tensor,
            position_ids=position_ids,
        )
        position_embeddings = self._compute_position_embeddings(hidden_states, position_ids)

        with torch.no_grad():
            continued = hidden_states
            for decoder_index in range(int(layer_idx), int(self.num_layers)):
                decoder_layer = backbone.layers[decoder_index]
                layer_attention_mask = self._select_layer_attention_mask(causal_masks, decoder_index)
                layer_output = decoder_layer(
                    continued,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                if isinstance(layer_output, tuple):
                    continued = layer_output[0]
                else:
                    continued = layer_output
            final_hidden = continued if int(layer_idx) >= int(self.num_layers) else backbone.norm(continued)
            mean_hidden = self._mean_pool_hidden(final_hidden, attention_mask_tensor)
            last_hidden = self._gather_last_token_hidden(final_hidden, attention_mask_tensor)
            last_logits = self.project_hidden_to_logits(last_hidden)
        return SequenceRepresentationBatch(
            mean_hidden=mean_hidden.detach().cpu().numpy().astype(np.float32),
            last_hidden=last_hidden.detach().cpu().numpy().astype(np.float32),
            last_logits=self._to_numpy_float32(last_logits),
            attention_mask=attention_mask_tensor.detach().cpu().numpy().astype(np.int32)
            if attention_mask_tensor is not None else None,
        )

    def get_forward_signatures(
        self,
        texts: list[str],
        *,
        layers: list[int],
        max_length: int | None = None,
        top_k: int = 8,
    ) -> ForwardSignatureBatch:
        torch = self._torch
        with torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
            logits = outputs.logits.to(dtype=torch.float32)
            attention_mask = encoded.get("attention_mask")
            last_logits = self._gather_last_token_logits(logits, attention_mask)
            last_probs = torch.nn.functional.softmax(last_logits, dim=-1)
            k = min(max(int(top_k), 1), int(last_probs.shape[-1]))
            topk_probs, topk_ids = torch.topk(last_probs, k=k, dim=-1)
            sorted_probs = torch.sort(last_probs, dim=-1, descending=True).values
            top1_prob = sorted_probs[:, 0]
            top2_prob = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else torch.zeros_like(top1_prob)
            last_token_entropy = self._entropy(last_probs)
            sequence_mean_entropy = self._sequence_mean_entropy(logits, attention_mask)
            global_features = {
                "last_token_entropy": last_token_entropy.detach().cpu().numpy().astype(np.float32),
                "sequence_mean_entropy": sequence_mean_entropy.detach().cpu().numpy().astype(np.float32),
                "last_token_top1_prob": top1_prob.detach().cpu().numpy().astype(np.float32),
                "last_token_margin": (top1_prob - top2_prob).detach().cpu().numpy().astype(np.float32),
                "last_token_topk_mass": topk_probs.sum(dim=-1).detach().cpu().numpy().astype(np.float32),
            }
            layer_features: dict[int, dict[str, np.ndarray]] = {}
            for layer_idx in layers:
                attention_tensor = self._attention_tensor_for_layer(outputs.attentions, layer_idx)
                entropy, peak_frac = self._attention_signature(attention_tensor, attention_mask)
                layer_features[int(layer_idx)] = {
                    "attention_entropy": entropy.detach().cpu().numpy().astype(np.float32),
                    "attention_peak_frac": peak_frac.detach().cpu().numpy().astype(np.float32),
                }
        return ForwardSignatureBatch(
            global_features=global_features,
            layer_features=layer_features,
            topk_token_ids=topk_ids.detach().cpu().numpy().astype(np.int32),
            topk_token_probs=topk_probs.detach().cpu().numpy().astype(np.float32),
        )

    def compute_sequence_losses(self, texts: list[str], *, max_length: int | None = None) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            encoded = self.tokenize(texts, max_length=max_length)
            outputs = self.model(**encoded, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            labels = encoded["input_ids"][:, 1:]
            loss_per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            mask = encoded.get("attention_mask")
            if mask is None:
                token_mask = torch.ones_like(labels, dtype=loss_per_token.dtype)
            else:
                token_mask = mask[:, 1:].to(loss_per_token.dtype)
            denom = token_mask.sum(dim=1).clamp_min(1.0)
            sequence_losses = (loss_per_token * token_mask).sum(dim=1) / denom
        return sequence_losses.detach().cpu().numpy().astype(np.float32)

    def get_gradient_statistics(
        self,
        texts: list[str],
        *,
        layer_idx: int,
        max_length: int | None = None,
    ) -> GradientBatchStatistics:
        torch = self._torch
        if layer_idx < 0 or layer_idx > self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}], got {layer_idx}")
        encoded = self.tokenize(texts, max_length=max_length)
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[layer_idx]
        hidden.retain_grad()
        logits = outputs.logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        mask = encoded.get("attention_mask")
        if mask is None:
            token_mask = torch.ones_like(labels, dtype=loss_per_token.dtype)
        else:
            token_mask = mask[:, 1:].to(loss_per_token.dtype)
        sequence_den = token_mask.sum(dim=1).clamp_min(1.0)
        sequence_losses = (loss_per_token * token_mask).sum(dim=1) / sequence_den
        loss = sequence_losses.mean()
        loss.backward()

        hidden_used = hidden[:, :-1, :].to(dtype=torch.float32)
        grad_used = hidden.grad[:, :-1, :].to(dtype=torch.float32)
        flat_mask = token_mask.reshape(-1) > 0
        flat_hidden = hidden_used.reshape(-1, hidden_used.shape[-1])[flat_mask]
        flat_grad = grad_used.reshape(-1, grad_used.shape[-1])[flat_mask]
        if flat_hidden.shape[0] == 0:
            dim = int(hidden_used.shape[-1])
            return GradientBatchStatistics(
                importance_sum=np.zeros((dim,), dtype=np.float32),
                covariance_numerator=np.zeros((dim, dim), dtype=np.float32),
                weight_denom=0.0,
                loss_sum=0.0,
                token_count=0,
                sequence_losses=np.zeros((len(texts),), dtype=np.float32),
            )

        importance_sum = flat_grad.abs().sum(dim=0)
        token_weights = flat_grad.pow(2).sum(dim=1).sqrt().clamp_min(1e-8)
        weighted_hidden = flat_hidden * token_weights.sqrt().unsqueeze(-1)
        covariance_numerator = weighted_hidden.T @ weighted_hidden
        loss_sum = float((loss_per_token * token_mask).sum().item())
        token_count = int(token_mask.sum().item())

        self.model.zero_grad(set_to_none=True)
        return GradientBatchStatistics(
            importance_sum=importance_sum.detach().cpu().numpy().astype(np.float32),
            covariance_numerator=covariance_numerator.detach().cpu().numpy().astype(np.float32),
            weight_denom=float(token_weights.sum().item()),
            loss_sum=loss_sum,
            token_count=token_count,
            sequence_losses=sequence_losses.detach().cpu().numpy().astype(np.float32),
        )

    def get_contrastive_statistics(
        self,
        left_texts: list[str],
        right_texts: list[str],
        *,
        layers: list[int],
        max_length: int | None = None,
    ) -> ContrastiveBatchStatistics:
        torch = self._torch
        if len(left_texts) != len(right_texts):
            raise ValueError("left_texts and right_texts must have the same batch size")
        if not left_texts:
            return ContrastiveBatchStatistics(
                js_divergence=np.zeros((0,), dtype=np.float32),
                layer_deltas={int(layer_idx): np.zeros((0, self.hidden_dim), dtype=np.float32) for layer_idx in layers},
            )

        with torch.no_grad():
            left_encoded = self.tokenize(left_texts, max_length=max_length)
            right_encoded = self.tokenize(right_texts, max_length=max_length)
            left_outputs = self.model(**left_encoded, output_hidden_states=True, use_cache=False)
            right_outputs = self.model(**right_encoded, output_hidden_states=True, use_cache=False)

            left_last = self._gather_last_token_logits(left_outputs.logits, left_encoded.get("attention_mask"))
            right_last = self._gather_last_token_logits(right_outputs.logits, right_encoded.get("attention_mask"))
            js_divergence = self._js_divergence(left_last, right_last)

            layer_deltas: dict[int, np.ndarray] = {}
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= len(left_outputs.hidden_states):
                    raise ValueError(f"layer_idx must be in [0, {len(left_outputs.hidden_states) - 1}], got {layer_idx}")
                left_hidden = self._mean_pool_hidden(left_outputs.hidden_states[layer_idx], left_encoded.get("attention_mask"))
                right_hidden = self._mean_pool_hidden(right_outputs.hidden_states[layer_idx], right_encoded.get("attention_mask"))
                delta = right_hidden - left_hidden
                layer_deltas[int(layer_idx)] = delta.detach().cpu().numpy().astype(np.float32)

        return ContrastiveBatchStatistics(
            js_divergence=js_divergence.detach().cpu().numpy().astype(np.float32),
            layer_deltas=layer_deltas,
        )

    def _mean_pool_hidden(self, hidden: Any, attention_mask: Any | None) -> Any:
        torch = self._torch
        hidden = hidden.to(dtype=torch.float32)
        if attention_mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=hidden.dtype, device=hidden.device)
        else:
            mask = attention_mask.to(dtype=torch.float32)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _to_torch_tensor(self, value: Any, *, dtype: Any | None = None) -> Any:
        torch = self._torch
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device)
        else:
            tensor = torch.as_tensor(value, device=self.device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _to_numpy_float32(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy(), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    def _default_position_ids(self, hidden_states: Any, attention_mask: Any | None) -> Any:
        torch = self._torch
        batch_size, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        base = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
        position_ids = base.expand(batch_size, seq_len)
        if attention_mask is None:
            return position_ids
        if attention_mask.shape != position_ids.shape:
            return position_ids
        return position_ids

    def _compute_position_embeddings(self, hidden_states: Any, position_ids: Any) -> Any:
        rotary_emb = self.model.model.rotary_emb
        try:
            return rotary_emb(hidden_states, position_ids=position_ids)
        except TypeError:
            return rotary_emb(hidden_states, position_ids)

    def _build_causal_masks(
        self,
        *,
        inputs_embeds: Any,
        attention_mask: Any | None,
        position_ids: Any,
    ) -> Any:
        try:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers.masking_utils is required for layer continuation") from exc

        config = self.model.config
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if getattr(self.model.model, "has_sliding_layers", False):
            return {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        if getattr(config, "sliding_window", None) is not None:
            return create_sliding_window_causal_mask(**mask_kwargs)
        return create_causal_mask(**mask_kwargs)

    def _select_layer_attention_mask(self, causal_masks: Any, decoder_index: int) -> Any:
        if isinstance(causal_masks, dict):
            layer_types = getattr(self.model.config, "layer_types", None)
            if layer_types is None:
                return causal_masks.get("full_attention")
            return causal_masks[str(layer_types[int(decoder_index)])]
        return causal_masks

    def _gather_last_token_logits(self, logits: Any, attention_mask: Any | None) -> Any:
        torch = self._torch
        logits = logits.to(dtype=torch.float32)
        if attention_mask is None:
            indices = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
        else:
            indices = attention_mask.sum(dim=1).to(dtype=torch.long).clamp_min(1) - 1
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        return logits[batch_idx, indices, :]

    def _gather_last_token_hidden(self, hidden: Any, attention_mask: Any | None) -> Any:
        torch = self._torch
        hidden = hidden.to(dtype=torch.float32)
        if attention_mask is None:
            indices = torch.full((hidden.shape[0],), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device)
        else:
            indices = attention_mask.sum(dim=1).to(dtype=torch.long).clamp_min(1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, indices, :]

    def _sequence_mean_entropy(self, logits: Any, attention_mask: Any | None) -> Any:
        torch = self._torch
        probs = torch.nn.functional.softmax(logits.to(dtype=torch.float32), dim=-1)
        entropy = self._entropy(probs)
        if attention_mask is None:
            mask = torch.ones_like(entropy, dtype=torch.float32)
        else:
            mask = attention_mask.to(dtype=torch.float32)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (entropy * mask).sum(dim=1) / denom

    def _entropy(self, probs: Any) -> Any:
        torch = self._torch
        probs = probs.to(dtype=torch.float32).clamp_min(1e-8)
        return -(probs * probs.log()).sum(dim=-1)

    def _attention_tensor_for_layer(self, attentions: Any, layer_idx: int) -> Any:
        if attentions is None:
            raise ValueError("Model did not return attentions")
        if layer_idx <= 0:
            index = 0
        else:
            index = layer_idx - 1
        if index < 0 or index >= len(attentions):
            raise ValueError(f"layer_idx must map to an attention layer in [0, {len(attentions)}), got {layer_idx}")
        return attentions[index].to(dtype=self._torch.float32)

    def _attention_signature(self, attention: Any, attention_mask: Any | None) -> tuple[Any, Any]:
        torch = self._torch
        if attention_mask is None:
            query_mask = torch.ones(attention.shape[0], attention.shape[2], dtype=torch.float32, device=attention.device)
        else:
            query_mask = attention_mask.to(dtype=torch.float32)
        probs = attention.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        peak = probs.max(dim=-1).values
        mean_entropy = entropy.mean(dim=1)
        mean_peak = peak.mean(dim=1)
        denom = query_mask.sum(dim=1).clamp_min(1.0)
        entropy_per_example = (mean_entropy * query_mask).sum(dim=1) / denom
        peak_per_example = (mean_peak * query_mask).sum(dim=1) / denom
        return entropy_per_example, peak_per_example

    def _js_divergence(self, left_logits: Any, right_logits: Any) -> Any:
        torch = self._torch
        left_logits = left_logits.to(dtype=torch.float32)
        right_logits = right_logits.to(dtype=torch.float32)
        left_log_probs = torch.nn.functional.log_softmax(left_logits, dim=-1)
        right_log_probs = torch.nn.functional.log_softmax(right_logits, dim=-1)
        left_probs = left_log_probs.exp()
        right_probs = right_log_probs.exp()
        mixture = (0.5 * (left_probs + right_probs)).clamp_min(1e-8)
        mixture_log = mixture.log()
        left_kl = (left_probs * (left_log_probs - mixture_log)).sum(dim=-1)
        right_kl = (right_probs * (right_log_probs - mixture_log)).sum(dim=-1)
        return torch.nan_to_num(0.5 * (left_kl + right_kl), nan=0.0, posinf=0.0, neginf=0.0)
