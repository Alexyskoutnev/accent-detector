from pathlib import Path
import logging
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed


class BucketDAO:
    def __init__(
        self,
        bucket: Any,  # Can be a mock or real S3 bucket object
        max_workers: int = 16,
        chunk_size: int = 100,
    ):
        self._bucket = bucket
        self.max_workers = max_workers  # tunable based on hardware constraints
        self.chunk_size = chunk_size  # tunable based on network constraints

    def _upload_files(self, files: list[tuple[str | Path, str]]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(files), self.chunk_size):
                chunk = files[i : i + self.chunk_size]
                futures = []
                for local_path, key in chunk:
                    future = executor.submit(self._upload_single, local_path, key)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Upload failed: {e}")

    def _upload_single(self, local_path: str | Path, key: str):
        try:
            self._bucket.upload_file(
                str(local_path), key
            )  # In production, this would be a call to the S3 API
        except Exception as e:
            logging.error(f"Failed to upload {local_path} to {key}: {e}")
            raise

    def _download_files(self, files: list[tuple[str, str | Path]]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(files), self.chunk_size):
                chunk = files[i : i + self.chunk_size]
                futures = []
                for key, local_path in chunk:
                    future = executor.submit(self._download_single, key, local_path)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Download failed: {e}")

    def _download_single(self, key: str, local_path: str | Path):
        try:
            self._bucket.download_file(
                key, str(local_path)
            )  # In production, this would be a call to the S3 API
        except Exception as e:
            logging.error(f"Failed to download {key} to {local_path}: {e}")
            raise

    def upload_dir(
        self, local_dir: str | Path, prefix: str = "", patterns: list[str] = None
    ):
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")
        upload_tasks = []
        if patterns:
            for pattern in patterns:
                for filepath in local_dir.rglob(pattern):
                    if filepath.is_file():
                        key = f"{prefix}/{filepath.relative_to(local_dir)}".lstrip("/")
                        upload_tasks.append((str(filepath), key))
        else:
            for filepath in local_dir.rglob("*"):
                if filepath.is_file():
                    key = f"{prefix}/{filepath.relative_to(local_dir)}".lstrip("/")
                    upload_tasks.append((str(filepath), key))
        self._upload_files(upload_tasks)

    def download_dir(
        self, prefix: str, local_dir: str | Path, patterns: list[str] = None
    ):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        response = self._bucket.list_objects(
            prefix=prefix
        )  # In production, this would be a call to the S3 API
        download_tasks = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if patterns and not any(key.endswith(pat.lstrip("*")) for pat in patterns):
                continue
            relative_path = key.replace(prefix, "").lstrip("/")
            target_path = local_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            download_tasks.append((key, target_path))
        self._download_files(download_tasks)

    def _get_object_content(self, key: str) -> bytes | None:
        try:
            response = self._bucket.get_object(
                key
            )  # In production, this would be a call to the S3 API
            content = response["Body"].read()
            response["Body"].close()
            return content
        except Exception as e:
            logging.error(f"Failed to get content for {key}: {e}")
            return None

    def read(self, keys: list[str]) -> Iterable[bytes]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(keys), self.chunk_size):
                chunk = keys[i : i + self.chunk_size]
                futures = []
                for key in chunk:
                    future = executor.submit(self._get_object_content, key)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        content = future.result()
                        if content:
                            yield content
                    except Exception as e:
                        logging.error(f"Content retrieval failed: {e}")
