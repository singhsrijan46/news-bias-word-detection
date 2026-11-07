"""
M3, the perspective model: one token model with three outputs per word.

    0  share of all annotators who would flag the word (as M2)
    1  share of liberal annotators     (ideology below -3, the paper's grouping)
    2  share of conservative annotators (ideology above 3)

Each output is trained with binary cross-entropy against its group's vote share,
weighted by how many annotators of that group judged the sentence. That is the same
as training on every annotator's own 0/1 label, grouped by ideology, so groups that
are small for a sentence count for little.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.annotations import annotator_ideology, ideology_groups, rebuild_sentence_annotations
from src.models.token_transformer import TokenModel, set_seed

HEADS = ("all", "liberal", "conservative")


@dataclass
class GroupTargets:
    """Per-word vote counts for each head, and how many annotators of each group judged the sentence."""

    votes: np.ndarray  # (words, 3)
    judges: np.ndarray  # (3,)

    def shares(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.judges > 0, self.votes / np.maximum(self.judges, 1), np.nan)


def build_group_targets(annotations, centre_band: int = 3) -> List[GroupTargets]:
    """Group targets for every sentence, in the same order as build_sentence_targets."""
    groups = ideology_groups(annotator_ideology(annotations), centre_band=centre_band)
    out = []
    for sentence in rebuild_sentence_annotations(annotations):
        votes = np.zeros((len(sentence.tokens), 3), dtype=np.float32)
        judges = np.zeros(3, dtype=np.float32)
        for j in sentence.judgements:
            heads = [0] + ([1] if groups.get(j.annotator_id) == "liberal" else
                           [2] if groups.get(j.annotator_id) == "conservative" else [])
            for h in heads:
                judges[h] += 1
                for p in j.positions:
                    votes[p, h] += 1
        out.append(GroupTargets(votes, judges))
    return out


class GroupTokenModel(TokenModel):
    """TokenModel with three outputs per word and annotator-count weights."""

    def _items(self, sentences, targets):
        items = super()._items(sentences, None)
        for item, group in zip(items, targets if targets is not None else [None] * len(items)):
            if group is not None:
                shares = np.nan_to_num(group.shares(), nan=0.0)
                item["targets"] = shares.astype(np.float32)
                item["weights"] = np.broadcast_to(group.judges, shares.shape).astype(np.float32)
        return items

    def _collate(self, batch):
        ids, mask, _ = super()._collate([{**b, "targets": None} for b in batch])
        width = ids.shape[1]
        labels = torch.zeros((len(batch), width, len(HEADS)))
        weights = torch.zeros((len(batch), width, len(HEADS)))
        for i, b in enumerate(batch):
            if b.get("targets") is None:
                continue
            keep = b["positions"] >= 0
            rows = torch.from_numpy(b["positions"][keep])
            labels[i, rows] = torch.from_numpy(b["targets"][keep])
            weights[i, rows] = torch.from_numpy(b["weights"][keep])
        return ids, mask, (labels, weights)

    def fit(self, sentences: Sequence, targets: Sequence[GroupTargets]) -> "GroupTokenModel":
        from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

        cfg = self.config
        set_seed(cfg.seed)
        model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, num_labels=len(HEADS)).to(self.device)
        loader = torch.utils.data.DataLoader(self._items(sentences, targets), batch_size=cfg.batch_size, shuffle=True,
                                             collate_fn=self._collate,
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
            for ids, mask, (labels, weights) in loader:
                ids, mask = ids.to(self.device), mask.to(self.device)
                labels, weights = labels.to(self.device), weights.to(self.device)
                with self._autocast():
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels, weight=weights, reduction="sum")
                loss = loss / weights.sum().clamp(min=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        self.model = model.eval()
        return self

    @torch.no_grad()
    def predict(self, sentences: Sequence, batch_size: int = 64) -> List[np.ndarray]:
        """Per-word probabilities, shape (words, 3): all, liberal, conservative."""
        items = self._items(sentences, None)
        out: List[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids, mask, _ = self._collate(batch)
            with self._autocast():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for i, b in enumerate(batch):
                p = np.full((len(b["positions"]), len(HEADS)), np.nan, dtype=np.float32)
                keep = b["positions"] >= 0
                p[keep] = probs[i, b["positions"][keep]]
                out.append(p)
        return out


def group_log_loss(votes: np.ndarray, judges: float, probs: np.ndarray) -> float:
    """Mean log loss per annotator judgement: how well probabilities explain one group's votes."""
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-(votes * np.log(p) + (judges - votes) * np.log(1 - p)).sum() / (judges * len(votes)))


def summarize_groups(group_targets: List[GroupTargets]) -> Dict[str, float]:
    judges = np.array([g.judges for g in group_targets])
    return {"sentences": len(group_targets),
            "mean_liberal_judges": float(judges[:, 1].mean()), "mean_conservative_judges": float(judges[:, 2].mean()),
            "sentences_with_both_groups": int(((judges[:, 1] > 0) & (judges[:, 2] > 0)).sum())}
