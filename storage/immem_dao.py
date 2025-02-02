from abc import ABC, abstractmethod
import dataclasses
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, BinaryIO, Protocol, Any
from pathlib import Path
import numpy as np
import hashlib
import threading
import logging
import json

class AudioStatus(Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclasses.dataclass
class AudioMetadata:
    file_id: str
    original_filename: str
    status: AudioStatus
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    samples: Optional[int] = None
    processing_date: datetime = dataclasses.field(default_factory=datetime.now)
    format: Optional[str] = None
    error_message: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'file_id': self.file_id,
            'original_filename': self.original_filename,
            'status': self.status.value,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'duration': self.duration,
            'file_size': self.file_size,
            'samples': self.samples,
            'processing_date': self.processing_date.isoformat(),
            'format': self.format,
            'error_message': self.error_message,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AudioMetadata':
        data = data.copy()
        data['status'] = AudioStatus(data['status'])
        data['processing_date'] = datetime.fromisoformat(data['processing_date'])
        return cls(**data)

class MockDao(Protocol):
    _id_to_obj : Dict[str, Any]

    def __init__(self, id_to_obj: Dict[str, Any]):
        self._id_to_obj = id_to_obj

    def insert(self, obj: list[Any]):
        self._id_to_obj[obj.file_id] = obj

    def read_by_id(self, file_id: str) -> Any:
        return self._id_to_obj.get(file_id)
    
    def read_all(self) -> List[Any]:
        return list(self._id_to_obj.values())
    
    def delete(self, file_id: str) -> bool:
        return self._id_to_obj.pop(file_id, None) is not None
        

@dataclasses.dataclass
class InMemAudioDAO:
    _data: Dict[str, AudioMetadata] = dataclasses.field(default_factory=dict)
    _lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    logger: logging.Logger = dataclasses.field(
        default_factory=lambda: logging.getLogger(__name__)
    )

    def insert(self, obj: AudioMetadata):
        with self._lock:
            if isinstance(obj, AudioMetadata):
                existing = self._data.get(obj.file_id)
                audio = existing.audio if existing else None
                self._data[obj.file_id] = obj
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

    def read_by_id(self, file_id: str) -> AudioMetadata | None:
        with self._lock:
            return self._data.get(file_id)