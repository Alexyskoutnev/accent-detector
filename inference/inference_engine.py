import dataclasses
import torch
from typing import Any, Iterable
from speechbrain.inference.interfaces import foreign_class


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
    model: Any = foreign_class(**ModelConfigs.to_dict()).to(device)

    def predict(
        self, audio_tensors: tuple[torch.Tensor | torch.Tensor]
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        out_prob, score, index, text_lab = self.model.classify_batch(*audio_tensors)
        yield out_prob, score, index, text_lab
