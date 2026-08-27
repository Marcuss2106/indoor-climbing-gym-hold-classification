import argparse
import sys
from pathlib import Path

from run_device import parse_yolo_device

PRETRAINED = Path("yolo11s.pt")
DATA_YAML = Path("climbing_gym.yaml")


def resolved_data_config():
	from download_dataset import download_dataset, named_dir

	datasets_dir = download_dataset()
	cfg = {
		"train": "train/images",
		"val": "valid/images",
		"nc": 6,
		"names": {
			0: "Crimp",
			1: "Jug",
			2: "Pinch",
			3: "Pocket",
			4: "Slope",
			5: "Volume",
		},
		"path": "Final_Dataset",
	}
	if DATA_YAML.exists():
		try:
			import yaml
			loaded = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
			if loaded:
				cfg.update(loaded)
		except Exception as exc:
			print(f"Could not read {DATA_YAML}, using built-in class names ({exc})", file=sys.stderr)

	folder_name = Path(str(cfg.get("path", "Final_Dataset"))).name
	final_dir = named_dir(datasets_dir, folder_name)
	if not final_dir.exists():
		print(f"Could not find {folder_name} under {datasets_dir}", file=sys.stderr)
		sys.exit(1)
	if isinstance(cfg.get("names"), dict):
		cfg["names"] = {int(k): v for k, v in cfg["names"].items()}
	# as_posix() keeps forward slashes on Windows, which Ultralytics accepts.
	cfg["path"] = final_dir.resolve().as_posix()
	return cfg


def main():
	parser = argparse.ArgumentParser(
		description="Train YOLOv11s hold detection (Windows and Linux)."
	)
	parser.add_argument(
		"load",
		nargs="?",
		choices=["load"],
		help="Start from a checkpoint instead of yolo11s.pt",
	)
	parser.add_argument(
		"weights",
		nargs="?",
		help="Weight file used with 'load', e.g. yolo11n.pt or best.pt",
	)
	parser.add_argument(
		"--device",
		default="auto",
		help="auto (default: CUDA GPU 0 if available, else CPU), or 0 / cpu / cuda:0",
	)
	args = parser.parse_args()

	data = resolved_data_config()
	print(f"Training YOLO on {data['path']}")

	weights = PRETRAINED
	if args.load == "load":
		if args.weights:
			weights = Path(args.weights)
		else:
			print("Give a valid checkpoint file!")
			print(f"Continuing from {PRETRAINED}.")

	device = parse_yolo_device(args.device)
	print(f"Using device: {device}")
	print(f"Starting weights: {weights}")

	from ultralytics import YOLO

	model = YOLO(str(weights))
	model.train(
		data=data,
		epochs=100,
		imgsz=640,
		batch=-1,
		multi_scale=True,
		amp=device != "cpu",
		cache=True,
		patience=5,
		name="yolov11s-climbing-gym",
		device=device,
	)


if __name__ == "__main__":
	main()
