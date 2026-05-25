"""Classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.labels import NUM_CLASSES


@dataclass(frozen=True, slots=True)
class MetricResult:
    accuracy: float
    top3_accuracy: float
    f1_macro: float
    f1_weighted: float
    loss: float
    confusion: tuple[tuple[int, ...], ...] | None = None


class MetricTracker:
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_samples = 0
        self.top3_correct = 0
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        for target, pred in zip(targets.view(-1), preds.view(-1), strict=True):
            self.confusion[int(target), int(pred)] += 1

    def update_logits(self, loss: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        preds = logits.argmax(dim=1)
        top3 = logits.topk(min(3, self.num_classes), dim=1).indices
        self.top3_correct += int((top3 == targets.unsqueeze(1)).any(dim=1).sum())

        for target, pred in zip(targets.view(-1), preds.view(-1), strict=True):
            self.confusion[int(target), int(pred)] += 1

    def compute(self) -> MetricResult:
        if self.total_samples == 0:
            return MetricResult(
                accuracy=0.0,
                top3_accuracy=0.0,
                f1_macro=0.0,
                f1_weighted=0.0,
                loss=0.0,
            )

        tp = self.confusion.diag().float()
        support = self.confusion.sum(dim=1).float()
        predicted = self.confusion.sum(dim=0).float()

        accuracy = float(tp.sum() / max(self.total_samples, 1))
        top3 = float(self.top3_correct / max(self.total_samples, 1))

        precision = tp / predicted.clamp_min(1.0)
        recall = tp / support.clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        valid = support > 0
        f1_macro = float(f1[valid].mean()) if valid.any() else 0.0
        f1_weighted = float((f1 * support).sum() / support.sum()) if support.sum() > 0 else 0.0

        return MetricResult(
            accuracy=accuracy,
            top3_accuracy=top3,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            loss=self.total_loss / self.total_samples,
            confusion=tuple(tuple(int(v) for v in row) for row in self.confusion.tolist()),
        )

