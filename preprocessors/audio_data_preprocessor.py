from typing import ClassVar, Generator
from pathlib import Path
import io
import os
import tempfile

import pandas as pd
import torchaudio
import torch

from storage.bucket_dao import BucketDAO


RAW_DATA_DIR_FROM_HUGGINGFACE = "raw_data"


class AudioDataPreprocessor:
    required_columns: ClassVar[tuple[str]] = (
        "filename",
        "file_missing?",
        "native_language",
    )
    csv_location: ClassVar[str] = f"{RAW_DATA_DIR_FROM_HUGGINGFACE}/speakers_all.csv"
    valid_audio_extension: ClassVar[tuple[str]] = ".mp3"
    audio_subdir: ClassVar[str] = (
        f"{RAW_DATA_DIR_FROM_HUGGINGFACE}/recordings/recordings"
    )
    batch_size: ClassVar[int] = 16 # tunable hyperparameter based on available memory or GPU
    target_sample_rate: ClassVar[int] = 16000 # tunable hyperparameter based on model requirements
    device: ClassVar[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _verify_csv(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = set(cls.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df

    @classmethod
    def _get_csv(cls, bucket_dao: BucketDAO) -> pd.DataFrame:
        csv_bytes = next(bucket_dao.read([cls.csv_location]))
        df = pd.read_csv(io.BytesIO(csv_bytes))
        return cls._verify_csv(df)

    @classmethod
    def _convert_batch_to_wav(
        cls, batch_bytes: list[bytes]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        results = []
        wav_lens = []

        for mp3_bytes in batch_bytes:
            with tempfile.NamedTemporaryFile(
                suffix=cls.valid_audio_extension, delete=False
            ) as tmp_file:
                tmp_file.write(mp3_bytes)
                tmp_path = tmp_file.name

            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=cls.target_sample_rate
                ).to(cls.device)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform = waveform.to(cls.device)
                if sample_rate != cls.target_sample_rate:
                    waveform = resampler(waveform)
                results.append(waveform.squeeze(0))
                wav_lens.append(waveform.shape[1])
            finally:
                os.remove(tmp_path)
        max_length = max(wav_lens)
        padded_waveforms = [
            torch.nn.functional.pad(w, (0, max_length - w.shape[0])) for w in results
        ]
        wav_tensor = torch.stack(padded_waveforms).to(cls.device)
        wav_lens_tensor = torch.tensor(
            [l / max_length for l in wav_lens], dtype=torch.float32
        ).to(cls.device)
        return wav_tensor, wav_lens_tensor

    @classmethod
    def _get_batch_audio(
        cls, bucket_dao: BucketDAO, file_names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_bytes = list(bucket_dao.read(file_names))
        return cls._convert_batch_to_wav(audio_bytes)

    @classmethod
    def _iterate_batches(
        cls, df: pd.DataFrame
    ) -> Generator[tuple[pd.DataFrame, list[str]], None, None]:
        batch_rows = []
        batch_paths = []
        for _, row in df.iterrows():
            if not row["file_missing?"]:
                full_path = (Path(cls.audio_subdir) / row["filename"]).with_suffix(
                    cls.valid_audio_extension
                )
                batch_rows.append(row)
                batch_paths.append(str(full_path))
                if len(batch_rows) >= cls.batch_size:
                    yield pd.DataFrame(batch_rows), batch_paths
                    batch_rows = []
                    batch_paths = []
        if batch_rows:
            yield pd.DataFrame(batch_rows), batch_paths

    @classmethod
    def process(
        cls, bucket_dao: BucketDAO
    ) -> Generator[
        tuple[tuple[torch.Tensor, torch.Tensor], list[pd.Series]], None, None
    ]:
        raw_csv_df = cls._get_csv(bucket_dao)
        for batch_rows, batch_paths in cls._iterate_batches(raw_csv_df):
            batch_tensor, batch_lens = cls._get_batch_audio(bucket_dao, batch_paths)
            yield (batch_tensor, batch_lens), batch_rows
