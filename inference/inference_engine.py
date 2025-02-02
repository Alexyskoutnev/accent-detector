import dataclasses
import torch
from typing import Any, Iterable, ClassVar
import logging
from speechbrain.inference.interfaces import foreign_class
from contextlib import contextmanager
import time


@contextmanager
def timer(name: str):
    """Context manager to measure time taken by a block of code and see our inference speed."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logging.info(f"{name} took {elapsed:.2f} seconds")


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
    source: str = "warisqr7/accent-id-commonaccent_xlsr-en-english"
    pymodule_file: str = "custom_interface.py"
    classname: str = "CustomEncoderWav2vec2Classifier"

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return dataclasses.asdict(cls())


@dataclasses.dataclass
class InferenceEngine:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: ClassVar[Any] = None

    def __post_init__(self):
        logging.debug(f"Initializing InferenceEngine on device: {self.device}")
        try:
            with timer("Model loading"):
                self.__class__.model = foreign_class(**ModelConfigs.to_dict()).to(
                    self.device
                )
            logging.debug("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def _validate_input(self, audio_tensors: tuple[torch.Tensor, torch.Tensor]):
        waveform, lengths = audio_tensors
        if not torch.isfinite(waveform).all():
            logging.error("Non-finite values detected in waveform")
            raise ValueError("Input waveform contains NaN or infinite values")
        if not (0 <= lengths).all() and (lengths <= 1).all():
            logging.error(
                f"Invalid length values detected: min={lengths.min()}, max={lengths.max()}"
            )
            raise ValueError("Lengths must be between 0 and 1")

    def _put_on_device(
        self, data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            with timer("Moving data to GPU if available"):
                result = tuple(d.to(self.device) for d in data)
            return result
        except Exception as e:
            logging.error(f"Failed to move data to device: {str(e)}")
            raise

    def _flag_low_model_confidence(
        self, score: torch.Tensor, score_threshold: float = 0.9
    ) -> torch.Tensor:
        try:
            # Create flags tensor (1 for low confidence, 0 for high confidence)
            confidence_flags = (score < score_threshold).to(torch.int)
            low_conf_mask = confidence_flags == 1
            if torch.any(low_conf_mask):
                low_conf_scores = score[low_conf_mask]
                logging.warning(
                    f"Low confidence scores detected: {low_conf_scores.tolist()}"
                )
            return confidence_flags
        except Exception as e:
            # as this is not vital, we can fallback to returning zeros
            logging.error(f"Error in confidence flagging: {str(e)}")
            return torch.zeros_like(score, dtype=torch.int)

    def _log_low_confidence_predictions(
        self, score: torch.Tensor, text_lab: str, confidence_flags: torch.Tensor
    ):
        """Log low confidence predictions to notify the engineer."""
        low_conf_count = confidence_flags.sum().item()
        if low_conf_count > 0:
            logging.warning(f"Found {low_conf_count} predictions with low confidence")
            low_conf_indices = torch.where(confidence_flags == 1)[0]
            for idx in low_conf_indices:
                logging.warning(
                    f"Sample {idx}: {text_lab[idx]} predicted with confidence {score[idx]:.4f}"
                )

    def predict(
        self, audio_tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]]:
        try:
            self._validate_input(audio_tensors)
            audio_tensors = self._put_on_device(audio_tensors)
            waveform, lengths = audio_tensors
            logging.info(
                f"Memory usage before inference: {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )
            with timer("Model inference"):
                # out_prob: All class probabilities (There are 16 classes)
                # score: Confidence score
                # index: Numerical class label
                # text_lab: Textual class label
                out_prob, score, index, text_lab = self.model.classify_batch(
                    *audio_tensors
                )
                confidence_flags = self._flag_low_model_confidence(score)
                low_conf_count = confidence_flags.sum().item()
                if low_conf_count > 0:
                    self._log_low_confidence_predictions(
                        score, text_lab, confidence_flags
                    )
            logging.info(
                f"Memory usage after inference: {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(
                    f"Memory usage after cleanup: {torch.cuda.memory_allocated()/1e9:.2f}GB"
                )
            yield out_prob, score, index, text_lab, confidence_flags
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
