import math
import numpy as np
import shutil
import sys

from pathlib import Path
from PIL import Image

from download_dataset import download_dataset, named_dir


def clamp(val, low, high):
    return max(low, min(high, val))

def clear_directory(dir_path: Path):
	"""Delete all files in the given directory. Replace images and labels subfolders."""
	if dir_path.exists() and dir_path.is_dir():
		for item in dir_path.iterdir():
			if item.is_file():
				item.unlink()
			elif item.is_dir():
				shutil.rmtree(item)
        # Re-create subdirectories
	dir_path.mkdir(exist_ok=True)
	(dir_path / "images").mkdir(parents=True, exist_ok=True)
	(dir_path / "labels").mkdir(parents=True, exist_ok=True)

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
    return mean.tolist(), std.tolist(), count


def calculate_mean_and_std(dataset_dir: Path, temp_dir: Path, resize: int):
	ann_files = sorted([p for p in (dataset_dir / "labels").iterdir() if p.suffix == ".txt"])
	total_saved = 0
	for ann in ann_files:
		total_saved += crop_and_save_from_ann(
			ann,
			dataset_dir / "images",
			temp_dir / "images",
			temp_dir / "labels",
			pad_fraction=0.1,
			tmp_fill_rgb=(255,255,255),
			output_size=resize,
			image_ext=".jpg",
			write_labels=False,   # label content not used for mean/std
        )
	print(f"Saved {total_saved} crops from {dataset_dir} to {temp_dir}")
	print(f"Computing dataset mean/std from {dataset_dir} images...")
	mean, std, count = compute_mean_std_from_folder(temp_dir / "images", image_ext="jpg")
	print(f"Clearing temporary directory {temp_dir}...")
	clear_directory(temp_dir)
	print(f"Computed mean (float 0..1): {mean}")
	print(f"Computed std  (float 0..1): {std}")
	return mean, std, count

def combine_mean_std(mean1, std1, n1, mean2, std2, n2):
	"""Combine two datasets' mean/std into overall mean/std."""
	mean1 = np.array(mean1)
	std1 = np.array(std1)
	mean2 = np.array(mean2)
	std2 = np.array(std2)

	overall_n = n1 + n2
	overall_mean = (mean1 * n1 + mean2 * n2) / overall_n

	var1 = std1 ** 2
	var2 = std2 ** 2

	overall_var = (
		(var1 + (mean1 - overall_mean) ** 2) * n1 +
		(var2 + (mean2 - overall_mean) ** 2) * n2
	) / overall_n

	overall_std = np.sqrt(overall_var)
	return overall_mean.tolist(), overall_std.tolist()

def pad_dataset(dataset_dir: Path, processed_dir: Path, resize: int, mean_rgb_int):
	ann_files = sorted([p for p in (dataset_dir / "labels").iterdir() if p.suffix == ".txt"])
	total_saved = 0
	for ann in ann_files:
		total_saved += crop_and_save_from_ann(
            ann,
            dataset_dir / "images",
            processed_dir / "images",
            processed_dir / "labels",
            pad_fraction=0.1,
            tmp_fill_rgb=mean_rgb_int,
            output_size=resize,
            image_ext=".jpg",
            write_labels=True,
        )
	print(f"Saved {total_saved} crops to {processed_dir / 'images'}")
     

# Example usage:
if __name__ == "__main__":
	datasets_dir = download_dataset()

	resize = 128
	processed_dir = Path("Processed_Dataset/")
	temp_dir = Path("temp_processing/")
	processed_dir.mkdir(exist_ok=True)
	temp_dir.mkdir(exist_ok=True)
	(processed_dir / "train/images").mkdir(parents=True, exist_ok=True)
	(processed_dir / "train/labels").mkdir(parents=True, exist_ok=True)
	(processed_dir / "test/images").mkdir(parents=True, exist_ok=True)
	(processed_dir / "test/labels").mkdir(parents=True, exist_ok=True)
	(processed_dir / "valid/images").mkdir(parents=True, exist_ok=True)
	(processed_dir / "valid/labels").mkdir(parents=True, exist_ok=True)
	(temp_dir / "images").mkdir(parents=True, exist_ok=True)
	(temp_dir / "labels").mkdir(parents=True, exist_ok=True)
      
	mean = [0.6681055533042859, 0.6137803857704651, 0.5396168694315177]
	mean_rgb_int = tuple(int(round(m * 255)) for m in mean)
	print(f"Mean RGB (int 0..255): {mean_rgb_int}")
    
	raw_dir = named_dir(datasets_dir, "Raw_dataset")
	synthetic_dir = named_dir(datasets_dir, "Synthetic_dataset")
	final_dir = named_dir(datasets_dir, "Final_Dataset")

	print("Padding Raw_dataset...")
	pad_dataset(
		raw_dir / "train",
		processed_dir / "train",
		resize,
		mean_rgb_int)
	pad_dataset(
		raw_dir / "test",
		processed_dir / "test",
		resize,
		mean_rgb_int)
      
	print("Padding Synthetic_dataset...")
	pad_dataset(
		synthetic_dir / "train",
		processed_dir / "train",
		resize,
		mean_rgb_int)
	pad_dataset(
		synthetic_dir / "test",
		processed_dir / "test",
		resize,
		mean_rgb_int)
      
	print("Padding Final_Dataset...")
	pad_dataset(
		final_dir / "train",
		processed_dir / "train",
		resize,
		mean_rgb_int)
	pad_dataset(
		dataset_dir=final_dir / "test",
		processed_dir=processed_dir / "test",
		resize=resize,
		mean_rgb_int=mean_rgb_int)
    
	print("Padding Validation sets")
	pad_dataset(
		raw_dir / "valid",
		processed_dir / "valid",
		resize,
		mean_rgb_int)
	pad_dataset(
		synthetic_dir / "valid",
		processed_dir / "valid",
		resize,
		mean_rgb_int)
	pad_dataset(
		final_dir / "valid",
		processed_dir / "valid",
		resize,
		mean_rgb_int)
      
	print("Calculating means and stds for datasets...")
	raw_mean, raw_std, raw_count = calculate_mean_and_std(
		raw_dir / "train",
		temp_dir,
		resize)
     
	synthetic_mean, synthetic_std, synthetic_count = calculate_mean_and_std(
		synthetic_dir / "train",
		temp_dir,
		resize)

	final_mean, final_std, final_count = calculate_mean_and_std(
		final_dir / "train",
		temp_dir,
        resize)
     
	mean, std = combine_mean_std(
		raw_mean, raw_std, raw_count,
		synthetic_mean, synthetic_std, synthetic_count)
	mean, std = combine_mean_std(
		mean, std, raw_count + synthetic_count,
		final_mean, final_std, final_count)
     
	print(f"Overall combined mean (float 0..1): {mean}")
	print(f"Overall combined std  (float 0..1): {std}")
	mean_rgb_int = tuple(int(round(m * 255)) for m in mean)
	print(f"Mean RGB (int 0..255): {mean_rgb_int}")