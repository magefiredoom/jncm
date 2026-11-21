"""GPT-2 for scoring without importing ``transformers`` (SPEC 6.1 Gen PPL).

Why this exists: ``import transformers`` (5.22.1) reads every ``.py`` file under
``transformers/`` to build its lazy import structure. On the cluster's Lustre ``/home``
that import took 15 min 16 s cold and 7 min 26 s warm, on every job (probe jobs 4746172
and 4748974, docs/STATE.md). The scorer needs only a tokenizer that returns ids and a model
that maps ids to logits, so this module provides both from ``tokenizers`` (which loads in
about a second) and plain torch. ``tests/test_gpt2_lean.py`` holds the Hugging Face classes
as the reference: logits agree to 1e-4 on a random model and to 5e-3 on gpt2-large.

Only the pieces GPT-2 needs are here: learned positions, pre-LayerNorm blocks, fused QKV
(``Conv1D`` layout, weights stored ``[in, out]``), ``gelu_new`` (tanh) MLP and a tied output
head. Dropout does not exist (inference only).
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a ``.safetensors`` file into CPU tensors, in file order (sequential reads).

    The format is an 8-byte little-endian header length, a JSON header mapping each name
    to ``dtype``, ``shape`` and ``data_offsets``, then the raw bytes.
    """
    tensors: dict[str, torch.Tensor] = {}
    with Path(path).open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header: dict[str, Any] = json.loads(handle.read(header_len))
        header.pop("__metadata__", None)
        base = 8 + header_len
        for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            start, end = info["data_offsets"]
            dtype = _DTYPES[info["dtype"]]
            handle.seek(base + start)
            buffer = bytearray(end - start)
            if end > start:
                handle.readinto(buffer)
                tensor = torch.frombuffer(buffer, dtype=dtype)
            else:
                tensor = torch.empty(0, dtype=dtype)
            tensors[name] = tensor.reshape(info["shape"])
    return tensors


class Conv1D(nn.Module):
    """GPT-2's linear layer: ``y = x @ weight + bias`` with ``weight`` of shape ``[in, out]``."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.bias = nn.Parameter(torch.empty(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError(f"n_embd {n_embd} is not divisible by n_head {n_head}")
        self.n_head = n_head
        self.c_attn = Conv1D(n_embd, 3 * n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: ``[B, T, D]``; mask: additive ``[B or 1, 1, T, T]``."""
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=mask)
        return self.c_proj(out.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = Conv1D(n_embd, 4 * n_embd)
        self.c_proj = Conv1D(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))  # gelu_new


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, eps: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = Attention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class LeanGPT2(nn.Module):
    """GPT-2 language model; ``forward`` returns an object with ``.logits`` ``[B, T, V]``."""

    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_positions = n_positions
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList(
            Block(n_embd, n_head, layer_norm_epsilon) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        """input_ids ``[B, T]`` (long); attention_mask ``[B, T]`` with 1 for real tokens.

        The mask is additive and finite (the dtype's minimum, not -inf), as in the
        Hugging Face model, so a fully padded row gives uniform attention instead of NaN.
        """
        batch, length = input_ids.shape
        if length > self.n_positions:
            raise ValueError(
                f"sequence length {length} exceeds the model's {self.n_positions} positions"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)
        floor = torch.finfo(x.dtype).min
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        mask = torch.zeros(1, 1, length, length, dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(~allowed, floor)
        if attention_mask is not None:
            keys = attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            mask = mask.expand(batch, 1, length, length).masked_fill(~keys, floor)
        for block in self.h:
            x = block(x, mask)
        x = self.ln_f(x)
        return SimpleNamespace(logits=x @ self.wte.weight.t())  # output head is tied to wte


def load_gpt2(path: str | Path, device: str | torch.device = "cpu") -> LeanGPT2:
    """Build a frozen, eval-mode GPT-2 from ``config.json`` and ``model.safetensors``."""
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("activation_function", "gelu_new") != "gelu_new":
        raise ValueError(f"unsupported activation {config['activation_function']!r}")
    model = LeanGPT2(
        vocab_size=config["vocab_size"],
        n_positions=config.get("n_positions", config.get("n_ctx", 1024)),
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-5),
    )
    state = read_safetensors(root / "model.safetensors")
    state = {
        (name[len("transformer.") :] if name.startswith("transformer.") else name): tensor
        for name, tensor in state.items()
    }
    state.pop("lm_head.weight", None)  # tied to wte
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"GPT-2 weights do not match the model: missing={missing}, "
                         f"unexpected={unexpected}")
    model.eval().requires_grad_(False)
    return model.to(device)


class LeanGPT2Tokenizer:
    """The GPT-2 BPE tokenizer from ``tokenizer.json``, via the ``tokenizers`` library.

    Mimics the part of the Hugging Face tokenizer interface the code uses: calling it
    returns ``{"input_ids": ...}`` (a list, or a ``[1, T]`` long tensor with
    ``return_tensors="pt"``), ``encode(text)`` returns the id list (``scripts/run_eval.py``
    and ``scripts/eval_data_row.py`` call it on ``scorer.tokenizer``), and
    ``eos_token_id`` / ``pad_token_id`` exist.
    """

    def __init__(self, path: str | Path, eos_token_id: int = 50256) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id: int | None = None  # GPT-2 has none; the scorer falls back to EOS

    def encode(self, text: str) -> list[int]:
        """Token ids of one text, never truncated, like the Hugging Face ``encode``."""
        return list(self._tokenizer.encode(text).ids)

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, Any]:
        ids = self.encode(text)
        if return_tensors is None:
            return {"input_ids": ids}
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        raise ValueError(f"return_tensors must be None or 'pt', got {return_tensors!r}")
