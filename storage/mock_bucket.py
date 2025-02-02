from typing import Protocol, BinaryIO, Any
from pathlib import Path
import shutil
import logging
import mimetypes
from datetime import datetime


class S3BucketProtocol(Protocol):
    """Protocol defining the S3 Bucket interface."""
    
    @property
    def name(self) -> str:
        ...

    def upload_file(self, 
                   filename: str | Path, 
                   key: str,
                   **kwargs) -> dict[str, Any]:
        ...

    def upload_fileobj(self,
                      fileobj: BinaryIO,
                      key: str,
                      **kwargs) -> dict[str, Any]:
        ...

    def download_file(self,
                     key: str,
                     filename: str | Path,
                     **kwargs) -> None:
        ...

    def download_fileobj(self,
                        key: str,
                        fileobj: BinaryIO) -> None:
        ...

    def get_object(self,
                  key: str,
                  **kwargs):
        ...

    def list_objects(self,
                    prefix: str = '',
                    **kwargs) -> dict[str, Any]:
        ...

class MockBucket(S3BucketProtocol):
    """Mock implementation of an S3 bucket."""    
    def __init__(self, 
                 base_path: str | Path):
        self.root_bucket_dir = Path(base_path)
        self._init_storage()
        
    @property
    def name(self) -> str:
        return self.root_bucket_dir.name

    def _init_storage(self) -> None:
        self.root_bucket_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_storage_path(self, key: str) -> Path:
        key = key.strip('/')
        return self.root_bucket_dir / key

    def upload_file(self,
                   filename: str | Path,
                   key: str) -> dict[str, Any]:
        try:
            with open(filename, 'rb') as f:
                return self.upload_fileobj(f, key)
        except Exception as e:
            logging.error(f"Error uploading file {filename}: {e}")
            raise

    def upload_fileobj(self,
                      fileobj: BinaryIO,
                      key: str,
                      ) -> dict[str, Any]:
        try:
            storage_path = self._get_storage_path(key)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(storage_path, 'wb') as f:
                shutil.copyfileobj(fileobj, f)
            return {
                'ETag': f'"{hash(str(storage_path))}"',
                'VersionId': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error uploading to {key}: {e}")
            raise

    def download_file(self,
                     key: str,
                     filename: str | Path) -> None:
        try:
            response = self.get_object(key)
            with open(filename, 'wb') as f:
                shutil.copyfileobj(response['Body'], f)
            response['Body'].close()
        except Exception as e:
            logging.error(f"Error downloading to {filename}: {e}")
            raise

    def download_fileobj(self,
                        key: str,
                        fileobj: BinaryIO):
        try:
            response = self.get_object(key)
            shutil.copyfileobj(response['Body'], fileobj)
            response['Body'].close()
        except Exception as e:
            logging.error(f"Error downloading to fileobj: {e}")
            raise

    def get_object(self,
                  key: str) -> dict[str, Any]:
        try:
            storage_path = self._get_storage_path(key)
            if not storage_path.exists():
                raise FileNotFoundError(f"Object not found: {storage_path}")
            content_type, _ = mimetypes.guess_type(str(storage_path))
            stats = storage_path.stat()
            return {
                'Body': open(storage_path, 'rb'),
                'ContentLength': stats.st_size,
                'ContentType': content_type or 'application/octet-stream',
                'ETag': f'"{hash(str(storage_path))}"',
                'LastModified': datetime.fromtimestamp(stats.st_mtime),
                'Metadata': {},
                'VersionId': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error getting object {key}: {e}")
            raise

    def list_objects(self,
                    prefix: str = '') -> dict[str, Any]:
        try:
            contents = []
            for file_path in self.root_bucket_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                key = str(file_path.relative_to(self.root_bucket_dir))
                if not key.startswith(prefix):
                    continue
                stats = file_path.stat()
                contents.append({
                    'Key': key,
                    'LastModified': datetime.fromtimestamp(stats.st_mtime),
                    'ETag': f'"{hash(str(file_path))}"',
                    'Size': stats.st_size,
                    'StorageClass': 'STANDARD'
                })
            result = {
                'Contents': contents,
                'Name': self.name,
                'Prefix': prefix,
            }
            return result
        except Exception as e:
            logging.error(f"Error listing objects: {e}")
            raise