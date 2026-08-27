import kagglehub
import os

from pathlib import Path

KAGGLE_DATASET = "diegospaziani/indoor-climbing-gym-hold-classification-dataset"


def download_dataset() -> Path:
	"""Download the Kaggle dataset into ./datasets and return the version directory."""
	os.environ["KAGGLEHUB_CACHE"] = str(Path.cwd())
	path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
	print(f"Dataset available at {path}")
	return path


def named_dir(parent: Path, name: str) -> Path:
	"""Return parent/name, matching an existing child case-insensitively."""
	if parent.exists():
		for child in parent.iterdir():
			if child.is_dir() and child.name.casefold() == name.casefold():
				return child
	return parent / name


def main():
	download_dataset()


if __name__ == "__main__":
	main()
