import math
from pathlib import Path
from PIL import Image
import kagglehub
import numpy as np
import shutil
import sys
import glob
import os

# -----------------------
# Helpers
# -----------------------
def clamp(val, low, high):
    return max(low, min(high, val))

def resize_with_padding(img: Image.Image, size: int, fill_rgb=(255,255,255)):
    """Resize while preserving aspect ratio so the longest side == size,
       then pad centered to (size, size) using fill_rgb (tuple of ints)."""
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Zero-dimension image")
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), fill_rgb)
    left = (size - new_w) // 2
    top  = (size - new_h) // 2
    padded.paste(img_resized, (left, top))
    return padded

def crop_and_save_from_ann(
    ann_path: Path,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    pad_fraction: float,
    tmp_fill_rgb,
    output_size: int,
    image_ext: str = ".jpg",
    write_labels=True,
):
    """Process a single annotation file: crop all lines and save images and labels."""
    image_basename = ann_path.stem
    image_path = images_dir / (image_basename + image_ext)
    if not image_path.exists():
        return 0

    with ann_path.open("r") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
    if not lines:
        return 0

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    saved_count = 0

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            print(f"[skip] bad line in {ann_path.name}: {line}", file=sys.stderr)
            continue
        cls = parts[0]
        try:
            x_c_n, y_c_n, w_n, h_n = map(float, parts[1:5])
        except ValueError:
            print(f"[skip] parse error in {ann_path.name}: {line}", file=sys.stderr)
            continue

        x_c = x_c_n * W
        y_c = y_c_n * H
        w_px = w_n * W
        h_px = h_n * H

        x1 = x_c - w_px / 2.0
        y1 = y_c - h_px / 2.0
        x2 = x_c + w_px / 2.0
        y2 = y_c + h_px / 2.0

        pad_px = int(max(w_px, h_px) * pad_fraction)
        x1p = clamp(math.floor(x1) - pad_px, 0, W - 1)
        y1p = clamp(math.floor(y1) - pad_px, 0, H - 1)
        x2p = clamp(math.ceil(x2) + pad_px, 0, W)
        y2p = clamp(math.ceil(y2) + pad_px, 0, H)

        crop_w = x2p - x1p
        crop_h = y2p - y1p
        if crop_w <= 0 or crop_h <= 0:
            print(f"[skip] invalid crop {ann_path.name} line {i}", file=sys.stderr)
            continue

        crop = img.crop((x1p, y1p, x2p, y2p))
        # resize+pad with chosen fill color
        crop_final = resize_with_padding(crop, output_size, fill_rgb=tmp_fill_rgb)

        out_img_name = f"{image_basename}_inst{i}{image_ext}"
        out_img_path = out_images_dir / out_img_name
        crop_final.save(out_img_path, quality=95)

        if write_labels:
            out_label_name = f"{image_basename}_inst{i}.txt"
            out_label_path = out_labels_dir / out_label_name
            # For classification we only write the class index on one line
            with out_label_path.open("w") as ol:
                ol.write(f"{cls}\n")

        saved_count += 1

    return saved_count

def compute_mean_std_from_folder(images_dir: Path, image_ext="jpg"):
    """Compute per-channel mean and std in [0,1] float for all images in folder."""
    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    count = 0
    files = list(images_dir.glob(f"*.{image_ext}"))
    if not files:
        raise RuntimeError(f"No images found in {images_dir}")
    for p in files:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        h, w, c = img.shape
        px = img.reshape(-1, 3)
        sums += px.sum(axis=0)
        sums_sq += (px ** 2).sum(axis=0)
        count += px.shape[0]
    mean = sums / count
    var = (sums_sq / count) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()


# Example usage:
if __name__ == "__main__":
	os.environ["KAGGLEHUB_CACHE"] = str(Path.cwd())
	path = kagglehub.dataset_download("diegospaziani/indoor-climbing-gym-hold-classification-dataset")
	# print("Processing training set from Raw Dataset...")
	# split_yolo_annotations_to_crops(
    #     annotations_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Raw_Dataset/train/labels",
    #     images_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Raw_Dataset/train/images",
    #     out_images_dir="Processed_Dataset/train/images",
    #     out_labels_dir="Processed_Dataset/train/labels",
    #     pad=0.1,
    #     image_ext=".jpg",
    # )
	# print("Processing test set from Raw Dataset...")
	# split_yolo_annotations_to_crops(
    #     annotations_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Raw_Dataset/test/labels",
    #     images_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Raw_Dataset/test/images",
    #     out_images_dir="Processed_Dataset/test/images",
    #     out_labels_dir="Processed_Dataset/test/labels",
    #     pad=0.1,
    #     image_ext=".jpg",
    # )
	# print("Processing training set from Synthetic Dataset...")
	# split_yolo_annotations_to_crops(
	# 	annotations_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Synthetic_Dataset/train/labels",
	# 	images_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Synthetic_Dataset/train/images",
	# 	out_images_dir="Processed_Dataset/train/images",
	# 	out_labels_dir="Processed_Dataset/train/labels",
	# 	pad=0.1,
	# 	image_ext=".jpg",
	# )
	# print("Processing test set from Synthetic Dataset...")
	# split_yolo_annotations_to_crops(
	# 	annotations_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Synthetic_Dataset/test/labels",
	# 	images_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Synthetic_Dataset/test/images",
	# 	out_images_dir="Processed_Dataset/test/images",
	# 	out_labels_dir="Processed_Dataset/test/labels",
	# 	pad=0.1,
	# 	image_ext=".jpg",
	# )
	# print("Processing training set from Final Dataset...")
	# split_yolo_annotations_to_crops(
	# 	annotations_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Final_Dataset/train/labels",
	# 	images_dir="datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/Final_Dataset/train/images",
	# 	out_images_dir="Processed_Dataset/train/images",
	# 	out_labels_dir="Processed_Dataset/train/labels",
	# 	pad=0.1,
	# 	image_ext=".jpg",
	# )
    
	resize = 128
	datasets_dir = Path("datasets/diegospaziani/indoor-climbing-gym-hold-classification-dataset/versions/3/")
	processed_dir = Path("Processed_Dataset/")
	temp_dir = Path("temp_processing/")
	processed_dir.mkdir(exist_ok=True)
	temp_dir.mkdir(exist_ok=True)
	ann_files = sorted([p for p in (datasets_dir / "Final_Dataset/test/labels").iterdir() if p.suffix == ".txt"])
	print("Processing test set from Final Dataset...")
	total_saved = 0
	for ann in ann_files:
		total_saved += crop_and_save_from_ann(
			ann,
			datasets_dir / "Final_Dataset/test/images",
			temp_dir / "images",
			temp_dir / "labels",
			pad_fraction=0.1,
			tmp_fill_rgb=(255,255,255),
			output_size=resize,
			image_ext=".jpg",
			write_labels=False,   # label content not used for mean/std
        )
	print(f"PASS 1 done — saved {total_saved} crops to {temp_dir}")
    # ----------------
    # Compute mean & std from PASS1 images
    # ----------------
	print("Computing dataset mean/std from pass1 images...")
	mean, std = compute_mean_std_from_folder(temp_dir / "images", image_ext=".jpg")
	mean_rgb_int = tuple(int(round(m * 255)) for m in mean)
	print(f"Computed mean (float 0..1): {mean}")
	print(f"Computed std  (float 0..1): {std}")
	print(f"Use mean color (integers) for padding: {mean_rgb_int}")

    # ----------------
    # PASS 2: crop + resize + pad with dataset mean -> final outputs
    # ----------------
	print("PASS 2: creating final images padded with dataset mean ...")
	total_saved2 = 0
	for ann in ann_files:
		total_saved2 += crop_and_save_from_ann(
            ann,
            datasets_dir / "Final_Dataset/test/images",
            processed_dir / "test/images",
            processed_dir / "test/labels",
            pad_fraction=0.1,
            tmp_fill_rgb=mean_rgb_int,
            output_size=resize,
            image_ext=".jpg",
            write_labels=True,
        )
	print(f"PASS 2 done — saved {total_saved2} final crops to {processed_dir / 'test/images'}")
	print("Done processing.")