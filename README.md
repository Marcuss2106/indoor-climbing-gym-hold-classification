# Indoor Climbing Hold Classification

This project contains two independent experiments on the same dataset. Scripts use `pathlib` and auto-select CUDA GPU 0 or CPU, so they run on Windows and Linux.

	Diego Spaziani. (2025). Indoor Climbing Gym Hold Classification Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/13795579

- **Classification:** a convolutional neural network that predicts hold type from cropped images of individual holds
- **Detection:** a YOLOv11 model that finds and labels holds on full wall photos and video

| Class ID | Hold Type |
| -------- | --------- |
| 0        | Crimp     |
| 1        | Jug       |
| 2        | Pinch     |
| 3        | Pocket    |
| 4        | Slope     |
| 5        | Volume    |

## Setup

Python 3.13 or newer. From the repo root:

```text
uv sync
```

Or: `pip install -e .`

Then run scripts with `python` (Windows and Linux) or `uv run python`. GPU is used automatically when CUDA is available; otherwise everything falls back to CPU.

The notebook (`analysis.ipynb`) uses sample crops, sample CNN checkpoints, YOLO plots, and `sample_run/` — you can open it without downloading the full Kaggle dataset.

Dataset download (needed to train, or to run `process_dataset.py` / `test_model.py` on the full splits) uses [KaggleHub](https://github.com/Kaggle/kagglehub). Sign in to Kaggle and place API credentials where KaggleHub expects them (typically `~/.kaggle/kaggle.json` on Linux, or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).

## Dataset

Each image is labeled as one of six hold classes: Crimp, Jug, Pinch, Pocket, Slope, and Volume. Annotations are YOLO-style TXT files. Sources include:

- Real photographs of climbing holds taken in an actual gym and manually annotated.
- Synthetic images of holds collected online and digitally edited to appear mounted on climbing walls.

`download_dataset.py` caches the Kaggle dataset under `datasets/` (gitignored). Classification and detection share that cache. Folder names such as `Final_Dataset` are matched case-insensitively so Linux matches a Windows-created layout.

### Checked-in samples

| Path | What it is |
| ---- | ---------- |
| `Sample_Processed_Dataset/` | Small cropped hold set for the classification half of the notebook |
| `sample_checkpoint.pth`, `sample_checkpoints/` | Sample CNN weights |
| `samples/` | Full-wall image/video for YOLO inference |
| `sample_run/` | Annotated YOLO output (still + video) from a previous predict pass |
| `runs/detect/yolov11s-climbing-gym/` | Trained YOLOv11s run (`best.pt` for inference, `last.pt` last epoch, plots) |
| `yolo11s.pt` | Pretrained YOLOv11s weights used by `train_yolo.py` |
| `yolo11n.pt` | Optional smaller nano model: `python train_yolo.py load yolo11n.pt` |

## Classification

`process_dataset.py` crops images according to YOLO labels, pads them to 128x128, and writes each crop plus a class label into `Processed_Dataset/`.

`train_model.py` trains a new CNN or resumes from a saved checkpoint.

`test_model.py` loads `checkpoint.pth` and evaluates on `Processed_Dataset/test`.

The model is a simple CNN defined in `Net.py`, using:
- Convolutional layers
- ReLU activations
- Max pooling
- Fully connected classification layer

Loss function:
- CrossEntropyLoss with class-weighted loss to handle class imbalance

## Detection

`climbing_gym.yaml` describes the YOLO class names and the `Final_Dataset` split. Prefer `train_yolo.py` over calling Ultralytics with that yaml alone: the script downloads the dataset and resolves the real folder path (including version and capitalization).

`train_yolo.py` trains YOLOv11s on full-wall images. A trained checkpoint is checked in at `runs/detect/yolov11s-climbing-gym/weights/best.pt`. Retraining writes a **new** folder under `runs/detect/` (gitignored) so the checked-in run is not overwritten.

`predict_yolo.py` runs detection on an image or video. `--device auto` is the default (GPU if present, else CPU).

## Use

#### Install
>uv sync
#### Download Dataset
>python download_dataset.py
#### Run Classification Preprocessing
>python process_dataset.py
#### Train Classification Model
>python train_model.py
#### Test Classification Model
>python test_model.py
#### Train YOLO Detector
>python train_yolo.py
#### Run YOLO Inference
>python predict_yolo.py
>python predict_yolo.py samples/videos/magnus_sample.mp4
>python predict_yolo.py samples/images/sample_image.webp --device cpu
#### Launch Notebook
>jupyter notebook

From the repo root, paths in the commands above work on Windows and Linux.

## To resume from a checkpoint
>python train_model.py load <model_file_path>
>python train_yolo.py load <weights_file_path>

`train_yolo.py load` starts training from those weights (for example `yolo11n.pt` or `best.pt`). It is not a full Ultralytics resume of optimizer state. Optional: `python train_yolo.py --device cpu`
