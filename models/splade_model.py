#!/usr/bin/env python3
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class EncodeOutput:
    embeddings: torch.Tensor
    nonzero_counts: torch.Tensor
    max_nonzero: int
    mean_nonzero: float


class SpladeModelWrapper:
    def __init__(
        self,
        model_name: str = "bizreach-inc/light-splade-japanese-14M",
        device: str = "cuda",
        padding_policy: str = "longest",
    ):
        self.model_name = model_name
        self.padding_policy = padding_policy

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        LOGGER.info("Loading model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def encode_batch(self, texts: List[str], max_length: int) -> EncodeOutput:
        if self.padding_policy == "max_length":
            padding = "max_length"
        else:
            padding = True

        tokenized = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        outputs = self.model(**tokenized)
        logits = outputs.logits
        activations = torch.log1p(torch.relu(logits))
        attn = tokenized.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(logits[..., 0], device=logits.device)
        attn = attn.unsqueeze(-1)
        pooled, _ = torch.max(activations * attn, dim=1)

        nonzero = (pooled > 0).sum(dim=-1)
        max_nonzero = int(nonzero.max().item()) if nonzero.numel() else 0
        mean_nonzero = float(nonzero.float().mean().item()) if nonzero.numel() else 0.0

        pooled_cpu = pooled.detach().cpu()
        return EncodeOutput(
            embeddings=pooled_cpu,
            nonzero_counts=nonzero.detach().cpu(),
            max_nonzero=max_nonzero,
            mean_nonzero=mean_nonzero,
        )

    def device_stats(self) -> Dict[str, str]:
        stats: Dict[str, str] = {"device": str(self.device)}
        if self.device.type == "cuda":
            stats.update({
                "gpu_name": torch.cuda.get_device_name(self.device),
                "compute_capability": str(torch.cuda.get_device_capability(self.device)),
                "total_memory_bytes": str(torch.cuda.get_device_properties(self.device).total_memory),
            })
        return stats
