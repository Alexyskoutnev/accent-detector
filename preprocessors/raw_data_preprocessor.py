from typing import ClassVar, Generator
from pathlib import Path
import io
import os
import tempfile
import logging

import pandas as pd
import torchaudio
import torch

from storage.mock_bucket import MockBucket
from storage.bucket_dao import BucketDAO


RAW_DATA_DIR_FROM_HUGGINGFACE = 'raw_data'

class RawDataPreprocessor:
    required_columns: ClassVar[tuple[str]] = ('filename', 'file_missing?', 'native_language')
    csv_location: ClassVar[str] = f'{RAW_DATA_DIR_FROM_HUGGINGFACE}/speakers_all.csv'
    valid_audio_extension: ClassVar[tuple[str]] = '.mp3'
    audio_subdir: ClassVar[str] = f'{RAW_DATA_DIR_FROM_HUGGINGFACE}/recordings/recordings'
    batch_size: ClassVar[int] = 4
    target_sample_rate: ClassVar[int] = 16000
    device: ClassVar[str] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def _convert_batch_to_wav(cls, 
                            batch_bytes: list[bytes]
                            ) -> list[torch.Tensor]:
        results = []
        for mp3_bytes in batch_bytes:
            with tempfile.NamedTemporaryFile(suffix=cls.valid_audio_extension, delete=False) as tmp_file:
                tmp_file.write(mp3_bytes)
                tmp_path = tmp_file.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
                # Put on GPU if available
                # waveform = waveform
                if sample_rate != cls.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, cls.target_sample_rate)
                    # log
                    logging.info(f"Resampling {tmp_path} from {sample_rate} to {cls.target_sample_rate}")
                    waveform = resampler(waveform)
                    logging.info(f"Now shape is {waveform.shape}")
                    sample_rate = cls.target_sample_rate
                results.append(waveform)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return results

    @classmethod
    def _get_batch_audio(cls, 
                        bucket_dao: BucketDAO, 
                        file_names: list[str]
                        ) -> list[torch.Tensor]:
        audio_bytes = list(bucket_dao.read(file_names))
        return cls._convert_batch_to_wav(audio_bytes)

    @classmethod
    def _iterate_batches(cls, 
                        df: pd.DataFrame
                        ) -> Generator[tuple[list[pd.Series], list[str]], None, None]:
        batch_rows = []
        batch_paths = []
        for _, row in df.iterrows():
            if not row['file_missing?']:
                full_path = (Path(cls.audio_subdir) / row['filename']).with_suffix(cls.valid_audio_extension)
                batch_rows.append(row)
                batch_paths.append(str(full_path))
                if len(batch_rows) >= cls.batch_size:
                    yield batch_rows, batch_paths
                    batch_rows = []
                    batch_paths = []
        if batch_rows:
            yield batch_rows, batch_paths

    @classmethod
    def process(cls, 
               bucket_dao: BucketDAO
               ) -> Generator[tuple[torch.Tensor, pd.Series], None, None]:
        raw_csv_df = cls._get_csv(bucket_dao)
        for batch_rows, batch_paths in cls._iterate_batches(raw_csv_df):
            batch_tensor = cls._get_batch_audio(bucket_dao, batch_paths)
            yield batch_tensor, batch_rows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bucket = MockBucket("./mock_bucket")
    dao = BucketDAO(bucket)
    RawDataPreprocessor.process(dao)
    
