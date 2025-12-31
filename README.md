# Indoor Climbing Hold Classification

This project trains a convolutional neural network (CNN) to classify indoor climbing hold types from images from this dataset:

	Diego Spaziani. (2025). Indoor Climbing Gym Hold Classification Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/13795579

The model predicts one of six hold categories based on cropped images of individual holds.



| Class ID | Hold Type |
| -------- | --------- |
| 0        | Crimp     |
| 1        | Jug       |
| 2        | Pinch     |
| 3        | Pocket    |
| 4        | Slope     |
| 5        | Volume    |

## Project Structure
`
├── Processed_Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│   │   ├── images/
│   └── └── labels/
├── plots/
├── checkpoints/
├── ClimbingHoldDataset.py
├── Net.py
├── process_dataset.py
├── train_model.py
├── checkpoint.pth
├── analysis.ipynb
└── README.md

`process_dataset.py` downloads the dataset from Kaggle using Kagglehub, crops the images according to YOLOv12 labels, pads the images to 128x128, and gives each image a corresponding label file in `Processed_Dataset/`

### Dataset Format
Each image is labeled according to one of six hold classes: Crimp, Jug, Pinch, Pocket, Slope, and Volume. These categories reflect common grip types used in indoor climbing and route setting.

The dataset includes TXT annotations compatible with YOLOv12 and two main sources of images:

-Real photographs of climbing holds taken in an actual gym and manually annotated.

-Synthetic images of holds collected online and digitally edited to appear mounted on climbing walls.

## Model

The model is a simple CNN defined in Net.py, using:
-Convolutional layers
-ReLU activations
-Max pooling
-Fully connected classification layer

Loss function:
-CrossEntropyLoss with class-weighted loss to handle class imbalance

## Use
#### Run Preprocessing
>python process_dataset.py
#### Train Model
>python train_model.py
#### Launch Notebook
>jupyter notebook

## To resume from a checkpoint
>python train_model.py load <model_file_path>
