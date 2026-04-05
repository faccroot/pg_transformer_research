from __future__ import annotations

import sys
import types
import unittest

from tools.representation_learning.model_adapter import HFCausalLMAdapter


class _FakeParameter:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class _FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    pad_token = None

    @classmethod
    def from_pretrained(cls, model_id: str, trust_remote_code: bool = False):
        return cls()


class _FakeConfig:
    model_type = "fake"
    num_hidden_layers = 4
    hidden_size = 16
    attn_implementation = "sdpa"
    _attn_implementation = "sdpa"


class _FakeModel:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.to_device = None
        self.eval_called = False

    def to(self, device: str) -> None:
        self.to_device = device

    def eval(self) -> None:
        self.eval_called = True

    def parameters(self):
        return [_FakeParameter(5), _FakeParameter(7)]


class _FallbackAutoModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, **kwargs})
        if "attn_implementation" in kwargs:
            raise TypeError("unexpected keyword")
        return _FakeModel()


class _EagerAutoModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, **kwargs})
        return _FakeModel()


class _ImportFallbackAutoModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, **kwargs})
        if kwargs.get("trust_remote_code", False):
            raise ImportError("broken dynamic module")
        return _FakeModel()


class _ImportFallbackTokenizer(_FakeTokenizer):
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, trust_remote_code: bool = False):
        cls.calls.append({"model_id": model_id, "trust_remote_code": trust_remote_code})
        if trust_remote_code:
            raise ImportError("broken tokenizer remote code")
        return cls()


class _FakeTorchModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))


class ModelAdapterInitTests(unittest.TestCase):
    def _install_fake_modules(self, auto_model_cls, auto_tokenizer_cls=_FakeTokenizer) -> tuple[object | None, object | None]:
        prev_torch = sys.modules.get("torch")
        prev_transformers = sys.modules.get("transformers")
        fake_torch = _FakeTorchModule()
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = auto_tokenizer_cls
        fake_transformers.AutoModelForCausalLM = auto_model_cls
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_transformers
        return prev_torch, prev_transformers

    def _restore_fake_modules(self, prev_torch, prev_transformers) -> None:
        if prev_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = prev_torch
        if prev_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = prev_transformers

    def test_prefers_eager_attention_when_supported(self) -> None:
        _EagerAutoModel.calls = []
        prev_torch, prev_transformers = self._install_fake_modules(_EagerAutoModel)
        try:
            adapter = HFCausalLMAdapter("fake/model")
        finally:
            self._restore_fake_modules(prev_torch, prev_transformers)
        self.assertEqual(_EagerAutoModel.calls[0]["attn_implementation"], "eager")
        self.assertEqual(adapter.model.config.attn_implementation, "eager")
        self.assertEqual(adapter.model.config._attn_implementation, "eager")
        self.assertEqual(adapter.num_parameters, 12)

    def test_falls_back_when_attn_implementation_kwarg_is_unsupported(self) -> None:
        _FallbackAutoModel.calls = []
        prev_torch, prev_transformers = self._install_fake_modules(_FallbackAutoModel)
        try:
            adapter = HFCausalLMAdapter("fake/model")
        finally:
            self._restore_fake_modules(prev_torch, prev_transformers)
        self.assertEqual(len(_FallbackAutoModel.calls), 2)
        self.assertIn("attn_implementation", _FallbackAutoModel.calls[0])
        self.assertNotIn("attn_implementation", _FallbackAutoModel.calls[1])
        self.assertEqual(adapter.model.config.attn_implementation, "eager")
        self.assertEqual(adapter.model.config._attn_implementation, "eager")

    def test_falls_back_from_broken_remote_code_to_builtin_model(self) -> None:
        _ImportFallbackAutoModel.calls = []
        _ImportFallbackTokenizer.calls = []
        prev_torch, prev_transformers = self._install_fake_modules(
            _ImportFallbackAutoModel,
            auto_tokenizer_cls=_ImportFallbackTokenizer,
        )
        try:
            adapter = HFCausalLMAdapter("fake/model", trust_remote_code=True)
        finally:
            self._restore_fake_modules(prev_torch, prev_transformers)
        self.assertEqual(len(_ImportFallbackTokenizer.calls), 2)
        self.assertTrue(_ImportFallbackTokenizer.calls[0]["trust_remote_code"])
        self.assertFalse(_ImportFallbackTokenizer.calls[1]["trust_remote_code"])
        self.assertEqual(len(_ImportFallbackAutoModel.calls), 2)
        self.assertTrue(_ImportFallbackAutoModel.calls[0]["trust_remote_code"])
        self.assertFalse(_ImportFallbackAutoModel.calls[1]["trust_remote_code"])
        self.assertEqual(adapter.model.config.attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
