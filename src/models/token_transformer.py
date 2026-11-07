"""
Transformer token models for bias words: one probability per word.

M1 trains on the paper's hard label (at least 4 annotators highlighted the word);
M2 trains on the vote share (the fraction of annotators who did). Both use binary
cross-entropy on each word's first sub-word token. With 0/1 targets that is
ordinary classification; with shares as targets it fits the predicted
probability to the share, so the model learns how many readers would flag a word.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.word_targets import SentenceTargets, first_subword_positions


@dataclass
class TrainConfig:
    model_name: str = "FacebookAI/roberta-base"
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenModel:
    """Fine-tunes a token classifier with one output per token and predicts per-word probabilities."""

    def __init__(self, config: TrainConfig, device: Optional[str] = None):
        from transformers import AutoTokenizer

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        on_gpu = self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if on_gpu and torch.cuda.is_bf16_supported() else (
            torch.float16 if on_gpu else None)

    def _items(self, sentences: Sequence[SentenceTargets], targets: Optional[Sequence[np.ndarray]]):
        enc = self.tokenizer([s.text for s in sentences], truncation=True, max_length=self.config.max_length,
                             return_offsets_mapping=True)
        items = []
        for k, (s, ids, offsets) in enumerate(zip(sentences, enc["input_ids"], enc["offset_mapping"])):
            items.append({"input_ids": ids, "positions": first_subword_positions(s.spans, offsets),
                          "targets": None if targets is None else np.asarray(targets[k], dtype=np.float32)})
        return items

    def _collate(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -1.0)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
            if b["targets"] is not None:
                keep = b["positions"] >= 0
                labels[i, torch.from_numpy(b["positions"][keep])] = torch.from_numpy(b["targets"][keep])
        return ids, mask, labels

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def fit(self, sentences: Sequence[SentenceTargets], targets: Sequence[np.ndarray]) -> "TokenModel":
        """Train on per-word targets: 0/1 labels (M1) or vote shares (M2)."""
        from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

        cfg = self.config
        set_seed(cfg.seed)
        model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, num_labels=1).to(self.device)
        items = self._items(sentences, targets)
        loader = torch.utils.data.DataLoader(items, batch_size=cfg.batch_size, shuffle=True, collate_fn=self._collate,
                                             generator=torch.Generator().manual_seed(cfg.seed))
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
        total = cfg.epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total), total)
        model.train()
        for _ in range(cfg.epochs):
            for ids, mask, labels in loader:
                ids, mask, labels = ids.to(self.device), mask.to(self.device), labels.to(self.device)
                with self._autocast():
                    logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                keep = labels >= 0
                loss = F.binary_cross_entropy_with_logits(logits[keep].float(), labels[keep])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        self.model = model.eval()
        return self

    @torch.no_grad()
    def predict(self, sentences: Sequence[SentenceTargets], batch_size: int = 64) -> List[np.ndarray]:
        """Per-word probabilities for each sentence (NaN for words cut off by truncation)."""
        items = self._items(sentences, None)
        out: List[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids, mask, _ = self._collate(batch)
            with self._autocast():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            probs = torch.sigmoid(logits.squeeze(-1).float()).cpu().numpy()
            for i, b in enumerate(batch):
                p = np.full(len(b["positions"]), np.nan, dtype=np.float32)
                keep = b["positions"] >= 0
                p[keep] = probs[i, b["positions"][keep]]
                out.append(p)
        return out

    def release(self) -> None:
        """Free GPU memory between folds."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
