"""Pick a compute device that works on Windows and Linux, with or without CUDA."""


def yolo_device():
	"""Return Ultralytics device: GPU 0 if CUDA is available, otherwise CPU."""
	try:
		import torch
		if torch.cuda.is_available():
			return 0
	except ImportError:
		pass
	return "cpu"


def parse_yolo_device(value: str | None):
	"""Interpret CLI/notebook device strings. ``auto`` (or empty) uses yolo_device()."""
	if value is None:
		return yolo_device()
	text = str(value).strip().lower()
	if text in ("", "auto"):
		return yolo_device()
	if text.isdigit():
		return int(text)
	return value
