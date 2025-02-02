from __future__ import annotations
import dataclasses
from typing import Tuple, ClassVar
import pandas as pd
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json


@dataclasses.dataclass(frozen=True)
class EvalStats:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float | str:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else "N/A"

    @property
    def recall(self) -> float | str:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else "N/A"

    @property
    def f1_score(self) -> float:
        denom = self.precision + self.recall
        return 2 * (self.precision * self.recall) / denom if denom > 0 else "N/A"

    def __str__(self) -> str:
        return (
            f"Total: {self.total}, Correct: {self.correct}, Incorrect: {self.incorrect}\n"
            f"Accuracy: {self.accuracy:.2%}\n"
            f"Precision: {self.precision:.2%}\n"
            f"Recall: {self.recall:.2%}\n"
            f"F1 Score: {self.f1_score:.2%}"
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class AccentEvaluator:
    """
    Evaluator for accent classification model.
    Uses class variables for global state to support parallel processing.
    """

    _running_stats: ClassVar[EvalStats] = EvalStats()
    target_accent: ClassVar[str] = "filipino"

    @classmethod
    def _evaluate_prediction(
        cls, prediction: Tuple[str, pd.Series]
    ) -> Tuple[int, int, int, int, int, int, int]:
        """
        Evaluates a single prediction for binary classification (Filipino vs. Non-Filipino)
        Logic Cases:
        1. True Positive (TP): Filipino accent correctly identified as Filipino
        - true_is_target = True
        - pred_is_target = True
        - is_correct = True
        2. True Negative (TN): Non-Filipino accent correctly identified as Non-Filipino
        - true_is_target = False
        - pred_is_target = False
        - is_correct = True
        3. False Positive (FP): Non-Filipino accent incorrectly identified as Filipino
        - true_is_target = False
        - pred_is_target = True
        - is_correct = False
        4. False Negative (FN): Filipino accent incorrectly identified as Non-Filipino
        - true_is_target = True
        - pred_is_target = False
        - is_correct = False
        """
        pred_label, true_row = prediction
        # Check if true label is Filipino
        true_is_target = cls._is_target_accent(true_row.filename)
        # Check if predicted label is Filipino
        pred_is_target = cls._is_target_accent(pred_label)
        is_correct = true_is_target == pred_is_target
        # return (total, correct, incorrect, true_positives, false_positives, true_negatives, false_negatives)
        return (
            1,  # count as one sample
            1 if is_correct else 0,  # Correct: predicted label matches true label
            (
                0 if is_correct else 1
            ),  # Incorrect: predicted label does not match true label
            (
                1 if is_correct and true_is_target else 0
            ),  # True Positive: correctly identified Filipino
            (
                1 if not is_correct and pred_is_target else 0
            ),  # False Positive: incorrectly identified Filipino
            (
                1 if is_correct and not true_is_target else 0
            ),  # True Negative: correctly identified Non-Filipino
            (
                1 if not is_correct and not pred_is_target else 0
            ),  # False Negative: incorrectly identified Non-Filipino
        )

    @classmethod
    def _is_target_accent(cls, file_name: str) -> bool:
        native_language = re.sub(r"[^a-zA-Z\s]", "", file_name)
        return native_language.lower() == cls.target_accent.lower()

    @classmethod
    def _create_batch_stats(
        cls, eval_results: list[tuple[int, int, int, int, int, int, int]]
    ) -> EvalStats:
        total = sum(r[0] for r in eval_results)
        correct = sum(r[1] for r in eval_results)
        return EvalStats(
            total=total,
            correct=correct,
            incorrect=sum(r[2] for r in eval_results),
            accuracy=correct / total if total > 0 else 0.0,
            true_positives=sum(r[3] for r in eval_results),
            false_positives=sum(r[4] for r in eval_results),
            true_negatives=sum(r[5] for r in eval_results),
            false_negatives=sum(r[6] for r in eval_results),
        )

    @classmethod
    def _combine_stats(cls, current: EvalStats, new: EvalStats) -> EvalStats:
        total = current.total + new.total
        correct = current.correct + new.correct
        return EvalStats(
            total=total,
            correct=correct,
            incorrect=current.incorrect + new.incorrect,
            accuracy=correct / total if total > 0 else 0.0,
            true_positives=current.true_positives + new.true_positives,
            false_positives=current.false_positives + new.false_positives,
            true_negatives=current.true_negatives + new.true_negatives,
            false_negatives=current.false_negatives + new.false_negatives,
        )

    def _place_on_cpu(
        self, model_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        return tuple(t.cpu() for t in model_outputs)

    @classmethod
    def evaluate(
        cls,
        model_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
        batch_metadata: pd.DataFrame,
    ) -> tuple[EvalStats, EvalStats]:
        """Evaluate model predictions against true labels."""
        # put on CPU if on GPU
        if torch.cuda.is_available():
            model_outputs = cls._place_on_cpu(model_outputs)
        _, _, _, predicted_labels = model_outputs
        prediction_pairs = zip(predicted_labels, batch_metadata.itertuples())
        eval_results = [cls._evaluate_prediction(pair) for pair in prediction_pairs]
        batch_stats = cls._create_batch_stats(eval_results)
        updated_stats = cls._combine_stats(cls._running_stats, batch_stats)
        cls._running_stats = updated_stats
        return batch_stats, updated_stats

    @classmethod
    def get_running_stats(cls) -> EvalStats:
        """Get running statistics for the evaluator."""
        return cls._running_stats

    @classmethod
    def reset_stats(cls):
        """Reset running statistics to zero."""
        cls._running_stats = EvalStats()


class ModelEvaluationHelpers:
    """Helper class to visualize evaluation metrics."""

    @staticmethod
    def plot_confusion_matrix(stats: EvalStats, save_path: str | None = None):
        matrix = [
            [stats.true_negatives, stats.false_positives],
            [stats.false_negatives, stats.true_positives],
        ]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Filipino", "Filipino"],
            yticklabels=["Not Filipino", "Filipino"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_metrics_history(
        metrics_history: list[EvalStats], save_path: str | None = None
    ):
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        values = {metric: [] for metric in metrics}
        for stat in metrics_history:
            for metric in metrics:
                values[metric].append(getattr(stat, metric))
        plt.figure(figsize=(10, 6))
        for metric, metric_values in values.items():
            plt.plot(metric_values, label=metric.replace("_", " ").title())
        plt.title("Metrics History")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
