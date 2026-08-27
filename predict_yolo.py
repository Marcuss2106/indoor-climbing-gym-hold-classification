import argparse
from pathlib import Path

from run_device import parse_yolo_device

DEFAULT_WEIGHTS = Path("runs/detect/yolov11s-climbing-gym/weights/best.pt")
DEFAULT_SOURCE = Path("samples/images/sample_image.webp")


def main():
	parser = argparse.ArgumentParser(
		description="Run YOLOv11 hold detection on an image or video (Windows and Linux)."
	)
	parser.add_argument(
		"source",
		nargs="?",
		default=str(DEFAULT_SOURCE),
		help="Image, video, or directory to run detection on",
	)
	parser.add_argument(
		"--weights",
		default=str(DEFAULT_WEIGHTS),
		help="Path to a YOLO checkpoint",
	)
	parser.add_argument(
		"--device",
		default="auto",
		help="auto (default: CUDA GPU 0 if available, else CPU), or 0 / cpu / cuda:0",
	)
	parser.add_argument("--conf", type=float, default=0.35)
	parser.add_argument(
		"--no-save",
		action="store_true",
		help="Do not write annotated results under runs/detect/",
	)
	args = parser.parse_args()

	device = parse_yolo_device(args.device)
	print(f"Using device: {device}")

	from ultralytics import YOLO

	model = YOLO(args.weights)
	model.predict(
		source=args.source,
		device=device,
		save=not args.no_save,
		show_labels=True,
		show_conf=False,
		conf=args.conf,
	)


if __name__ == "__main__":
	main()
