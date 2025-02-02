from __future__ import annotations
import dataclasses
from typing import Tuple, ClassVar
import pandas as pd
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json


@dataclasses.dataclass(frozen=False)
class EvalStats:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    low_confidence_count: int = 0 # number of low confidence predictions
    total_confidence_score: float = 0.0 # sum of all confidence scores
    
    @property
    def average_confidence_score(self) -> float:
        return self.total_confidence_score / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        if self.precision == 0 or self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def total_filipino_accents(self) -> int:
        return self.true_positives + self.false_negatives
    
    @property
    def total_non_filipino_accents(self) -> int:
        return self.true_negatives + self.false_positives
    
    @property
    def class_distribution(self) -> str:
        total = self.total
        if total == 0:
            return ""
        pos_ratio = (self.total_filipino_accents / total) * 100
        neg_ratio = (self.total_non_filipino_accents / total) * 100
        return (
            f"Class Distribution:\n"
            f"Filipino Accents: {self.total_filipino_accents}/{total} ({pos_ratio:.1f}%)\n"
            f"Non-Filipino Accents: {self.total_non_filipino_accents}/{total} ({neg_ratio:.1f}%)"
        )

    def __str__(self) -> str:
        base_str = (
            f"Total: {self.total}, Correct: {self.correct}, Incorrect: {self.incorrect}\n"
            f"Accuracy: {self.accuracy:.2%}\n"
            f"Average Confidence Score: {self.average_confidence_score:.3f}\n"
            f"Precision: {'N/A' if isinstance(self.precision, int) else f'{self.precision:.2%}'}\n"
            f"Recall: {'N/A' if isinstance(self.recall, int) else f'{self.recall:.2%}'}\n"
            f"F1 Score: {'N/A' if isinstance(self.f1_score, int) else f'{self.f1_score:.2%}'}"
            f"Class Distribution:\n"
            f"Filipino Accents: {self.total_filipino_accents}/{self.total}\n"
            f"Non-Filipino Accents: {self.total_non_filipino_accents}/{self.total}\n"
        )
        return base_str

    def to_dict(self) -> dict:
        base_dict = {
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
            "total_filipino_accent": self.total_filipino_accents,
            "total_non_filipino_accent": self.total_non_filipino_accents,
            "average_confidence_score": self.average_confidence_score, 
            "low_confidence_count": self.low_confidence_count,
        }
        return base_dict

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
        cls, prediction: Tuple[str, pd.Series, bool, float]
    ) -> Tuple[int, int, int, int, int, int, int, int, float]:
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
        pred_label, true_row, is_low_confidence, confidence_score = prediction
        # Check if true label is Filipino
        true_is_target = cls._is_target_accent(true_row.filename)
        # Check if predicted label is Filipino
        pred_is_target = cls._is_target_accent(pred_label)
        is_correct = true_is_target == pred_is_target
        # Update confidence level stats
        low_conf_count = 1 if is_low_confidence else 0
        # return (total, correct, incorrect, true_positives, false_positives, true_negatives, false_negatives, low_confidence_count, confidence_score
        return (
            1,  # total
            1 if is_correct else 0,  # correct
            0 if is_correct else 1,  # incorrect
            1 if is_correct and true_is_target else 0,  # true_positives
            1 if not is_correct and pred_is_target else 0,  # false_positives
            1 if is_correct and not true_is_target else 0,  # true_negatives
            1 if not is_correct and not pred_is_target else 0,  # false_negatives
            low_conf_count,  # low_confidence_count
            confidence_score,  # confidence_score
        )

    @classmethod
    def _is_target_accent(cls, file_name: str) -> bool:
        native_language = re.sub(r"[^a-zA-Z\s]", "", file_name)
        return native_language.lower() == cls.target_accent.lower()

    @classmethod
    def _create_batch_stats(cls, eval_results: list[tuple[int, int, int, int, int, int, int, int, float]]) -> EvalStats:
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
            low_confidence_count=sum(r[7] for r in eval_results),
            total_confidence_score=sum(r[8] for r in eval_results),
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
            low_confidence_count=current.low_confidence_count + new.low_confidence_count,
            total_confidence_score=current.total_confidence_score + new.total_confidence_score,
        )

    @classmethod
    def _place_on_cpu(
        cls,
        model_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        prob, score, index, labels, confidence_flags = model_outputs
        return (
            prob.cpu(),
            score.cpu(),
            index.cpu(),
            labels,
            confidence_flags.cpu(),
        )
    
    @classmethod
    def evaluate(
        cls,
        model_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor],
        batch_metadata: pd.DataFrame,
    ) -> tuple[EvalStats, EvalStats]:
        # put on CPU if on GPU
        if torch.cuda.is_available():
            model_outputs = cls._place_on_cpu(model_outputs)
        _, scores, _, predicted_labels, confidence_flags = model_outputs
        prediction_pairs = zip(
            predicted_labels, 
            batch_metadata.itertuples(),
            confidence_flags.bool().tolist(),
            scores.tolist()
        )
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
            plt.savefig(save_path, bbox_inches='tight')
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
        plt.title("Evaluation Metrics History")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_average_confidence_history(
        metrics_history: list[EvalStats], save_path: str | None = None
    ):
        confidence_scores = [stat.average_confidence_score for stat in metrics_history]
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_scores, label="Average Confidence Score")
        plt.title("Average Confidence Score History")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()