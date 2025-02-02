from pathlib import Path
import shutil
import json
import logging
from typing import BinaryIO, Any, Iterable
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import mimetypes

class MockBucket:
    # Mock S3 bucket class for testing purposes
    def __init__(self, base_path: str | Path):
        self.root_bucket_dir = Path(base_path)
        self._init_storage()
        
    @property
    def name(self) -> str:
        return self.root_bucket_dir.name

    def _init_storage(self):
        self.root_bucket_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_storage_path(self, key: str) -> Path:
        key = key.strip('/')
        return self.root_bucket_dir / key

    def upload_file(self, 
                   filename: str | Path, 
                   key: str, 
                   **kwargs) -> dict[str, Any]:
        try:
            with open(filename, 'rb') as f:
                return self.upload_fileobj(f, key, **kwargs)
        except Exception as e:
            logging.error(f"Error uploading file {filename}: {e}")
            raise

    def upload_fileobj(self, 
                      fileobj: BinaryIO, 
                      key: str, 
                      **kwargs) -> dict[str, Any]:
        return self._upload(key=key, body=fileobj, **kwargs)

    def download_file(self, 
                     key: str, 
                     filename: str | Path, 
                     **kwargs) -> None:
        try:
            response = self._download(key, **kwargs)
            with open(filename, 'wb') as f:
                shutil.copyfileobj(response['Body'], f)
        except Exception as e:
            logging.error(f"Error downloading file {filename}: {e}")
            raise

    def download_fileobj(self, 
                        key: str, 
                        fileobj: BinaryIO, 
                        **kwargs) -> None:
        try:
            response = self._download(key, **kwargs)
            shutil.copyfileobj(response['Body'], fileobj)
        except Exception as e:
            logging.error(f"Error downloading to fileobj: {e}")
            raise

    def _upload(self,
               key: str,
               body: bytes | str | BinaryIO | np.ndarray | dict,
               **kwargs) -> dict[str, Any]:
        try:
            storage_path = self._get_storage_path(key)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(body, 'read'):
                with open(storage_path, 'wb') as f:
                    shutil.copyfileobj(body, f)
            elif isinstance(body, bytes):
                with open(storage_path, 'wb') as f:
                    f.write(body)
            elif isinstance(body, str):
                with open(storage_path, 'w') as f:
                    f.write(body)
            elif isinstance(body, np.ndarray):
                storage_path = storage_path.with_suffix('.npy')
                np.save(storage_path, body)
            elif isinstance(body, dict):
                with open(storage_path, 'w') as f:
                    json.dump(body, f)
            else:
                raise ValueError(f"Unsupported body type: {type(body)}")
            return {
                'ETag': f'"{hash(str(storage_path))}"',
                'VersionId': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error uploading object: {e}")
            raise

    def _download(self, key: str, **kwargs) -> dict[str, Any]:
        try:
            storage_path = self._get_storage_path(key)
            if not storage_path.exists():
                if storage_path.with_suffix('.npy').exists():
                    storage_path = storage_path.with_suffix('.npy')
                else:
                    raise FileNotFoundError(f"Object not found: {storage_path}")
            content_type, _ = mimetypes.guess_type(str(storage_path))
            body = open(storage_path, 'rb')
            return {
                'Body': body,
                'ContentType': content_type,
                'ContentLength': storage_path.stat().st_size,
                'LastModified': datetime.fromtimestamp(storage_path.stat().st_mtime),
                'ETag': f'"{hash(str(storage_path))}"',
                'VersionId': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error downloading object: {e}")
            raise

    def list_objects(self,
                    prefix: str = '',
                    **kwargs) -> dict[str, Any]:
        try:
            files = []
            for file_path in self.root_bucket_dir.rglob('*'):
                if file_path.is_file():
                    key = str(file_path.relative_to(self.root_bucket_dir))
                    if key.startswith(prefix):
                        files.append({
                            'Key': key,
                            'LastModified': datetime.fromtimestamp(file_path.stat().st_mtime),
                            'Size': file_path.stat().st_size,
                            'StorageClass': 'STANDARD'
                        })
            return {
                'Contents': files,
                'Name': self.root_bucket_dir.name,
                'Prefix': prefix,
                'MaxKeys': 1000,
                'IsTruncated': False
            }
        except Exception as e:
            logging.error(f"Error listing objects: {e}")
            raise

    def delete_object(self,
                     key: str,
                     **kwargs) -> dict[str, Any]:
        try:
            storage_path = self._get_storage_path(key)
            if storage_path.exists():
                storage_path.unlink()
            elif storage_path.with_suffix('.npy').exists():
                storage_path.with_suffix('.npy').unlink()
                
            return {
                'DeleteMarker': True,
                'VersionId': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error deleting object: {e}")
            raise

class BucketDAO:
    # Interface for uploading and downloading files to/from a bucket
    def __init__(self, 
                 bucket: MockBucket, 
                 max_workers: int = 16,
                 chunk_size: int = 100,
                 parallel: bool = True):
        self.bucket = bucket
        self.max_workers = max_workers # Number of threads to use for parallel uploads/downloads (tuneable)
        self.chunk_size = chunk_size # Number of files to upload/download in parallel given hardware limitations (tuneable)
        self.parallel = parallel

    def upload_files(self, 
                     files: list[tuple[str|Path, str]]) -> list[dict[str, Any]]:
        results = []
        for i in range(0, len(files), self.chunk_size):
            chunk = files[i:i + self.chunk_size]
            if self.parallel and len(chunk) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    chunk_results = list(executor.map(self._upload_single, chunk))
                    results.extend(chunk_results)
            else:
                for file_info in chunk:
                    results.append(self._upload_single(file_info))
        return results
    
    def _upload_single(self, 
                       file_info: tuple[str | Path, str]) -> dict[str, Any]:
        filename, key = file_info
        try:
            result = self.bucket.upload_file(filename=filename, key=key) # In production this would be an S3 handler
            return {
                'key': key,
                'filename': str(filename),
                'success': True,
                'result': result
            }
        except Exception as e:
            logging.error(f"Failed to upload {filename}: {e}")
            return {
                'key': key,
                'filename': str(filename),
                'success': False,
                'error': str(e)
            }

    def download_files(
            self,
            files: list[tuple[str, str | Path]],
            parallel: bool = True
        ) -> Iterable[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(files), self.chunk_size):
                chunk = files[i:i + self.chunk_size]
                if parallel and len(chunk) > 1:
                        chunk_results = list(executor.map(self._download_single, chunk))
                        yield from chunk_results
                else:
                    for file_info in chunk:
                        yield self._download_single(file_info)

    def upload_dir(
        self,
        directory: str | Path,
        prefix: str = '',
        patterns: list[str] = None
    ) -> list[dict[str, Any]]:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        files = []
        if patterns:
            for pattern in patterns:
                files.extend(directory.rglob(pattern))
        else:
            files = list(directory.rglob('*'))
        upload_tasks = [(str(f), f"{prefix}/{f.relative_to(directory)}".lstrip('/')) 
            for f in files if f.is_file()
        ]
        return self.upload_files(upload_tasks)

    def download_dir(
                    self,
                    prefix: str,
                    target_dir: str | Path,
                    patterns: list[str] = None
                ) -> Iterable[dict[str, Any]]:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        objects = self.bucket.list_objects(prefix=prefix)
        files = []
        for obj in objects.get('Contents', []):
            key = obj['Key']
            if patterns and not any(key.endswith(pat.lstrip('*')) for pat in patterns):
                continue
            relative_path = key.replace(prefix, '').lstrip('/')
            target_path = target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            files.append((key, target_path))
        yield from self.download_files(files)

    def _download_single(self, 
                         file_info: tuple[str, str | Path]) -> dict[str, Any]:
        key, filename = file_info
        try:
            self.bucket.download_file(key=key, filename=filename)  # In production this would be an S3 handler
            return {
                'key': key, 
                'filename': str(filename), 
                'success': True
            }
        except Exception as e:
            logging.error(f"Failed to download {key} to {filename}: {e}")
            return {
                'key': key,
                'filename': str(filename),
                'success': False,
                'error': str(e)
            }