import kagglehub
from pathlib import Path
import shutil


class KaggleDatasetDownloader:

    def __init__(self, dataset_name: str = "rtatman/speech-accent-archive"):
        self.dataset_name = dataset_name

    def download_to_dir(self, target_dir: str, force: bool = False) -> Path:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        temp_path = kagglehub.dataset_download(self.dataset_name)
        self._move_files(Path(temp_path), target_path)
        return target_path

    def _move_files(self, source_path: Path, target_path: Path):
        for item in source_path.iterdir():
            target_item = target_path / item.name
            if target_item.exists():
                if target_item.is_file():
                    target_item.unlink()
                else:
                    shutil.rmtree(target_item)
            if item.is_file():
                shutil.copy2(item, target_item)
            else:
                shutil.copytree(item, target_item)


def download_kaggle_dataset(target_dir: str = "./", force: bool = False) -> Path:
    downloader = KaggleDatasetDownloader()
    return downloader.download_to_dir(target_dir, force)
