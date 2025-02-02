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
    """
    Preprocesses audio files for accent detection model training and inference.
    
    This preprocessor:
    - Loads and validates speaker metadata from CSV
    - Processes audio files in batches to manage memory efficiently
    - Converts audio to the required format for XLSR model:
        * 16kHz sample rate
        * Mono channel
        * Normalized and padded tensors
    """
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
    batch_size: ClassVar[int] = 4 # tunable hyperparameter based on available memory or GPU
    target_sample_rate: ClassVar[int] = 16000 # required by XLSR model
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
        """
        Converts a batch of MP3 files to WAV format with standardized parameters:
        - Resamples to 16kHz
        - Converts to mono channel
        - Pads to consistent length within batch
        - Normalizes lengths for XLSR model
        """
        results = []
        wav_lens = []
        for mp3_bytes in batch_bytes:
            with tempfile.NamedTemporaryFile(
                suffix=cls.valid_audio_extension, delete=False
            ) as tmp_file:
                tmp_file.write(mp3_bytes)
                tmp_path = tmp_file.name
            try:
                # We convert the mp3 to a wav tile to take advantage of batching processing for the xlsr model
                waveform, sample_rate = torchaudio.load(tmp_path)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=cls.target_sample_rate
                ).to(cls.device)
                # "warisqr7/accent-id-commonaccent_xlsr-en-english" model requires mono-channel audio and 16kHz sample rate
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform = waveform.to(cls.device)
                if sample_rate != cls.target_sample_rate:
                    waveform = resampler(waveform)
                results.append(waveform.squeeze(0))
                wav_lens.append(waveform.shape[1])
            finally:
                os.remove(tmp_path)
        # getting the max length of the waveforms in the batch
        max_length = max(wav_lens)
        # We pad the waveforms to the same length to create a batch
        padded_waveforms = [
            torch.nn.functional.pad(w, (0, max_length - w.shape[0])) for w in results
        ]
        wav_tensor = torch.stack(padded_waveforms).to(cls.device)
        # We normalize the lengths to be between 0 and 1 based on the max length in the batch (the xmlr model requires this)
        wav_lens_tensor = torch.tensor(
            [l / max_length for l in wav_lens], dtype=torch.float32
        ).to(cls.device)
        return wav_tensor, wav_lens_tensor

    @classmethod
    def _get_batch_audio(
        cls, bucket_dao: BucketDAO, file_names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # This allows us to read files at batch level (without consuming uncessary memory or reading the entire db) aka the size of file_names corresponds to the batch size
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
        """
        Main processing pipeline that:
        1. Loads speaker metadata
        2. Processes audio files in batches
        3. Yields processed tensors with corresponding metadata
        """
        raw_csv_df = cls._get_csv(bucket_dao)
        for batch_rows, batch_paths in cls._iterate_batches(raw_csv_df):
            batch_tensor, batch_lens = cls._get_batch_audio(bucket_dao, batch_paths)
            yield (batch_tensor, batch_lens), batch_rows
