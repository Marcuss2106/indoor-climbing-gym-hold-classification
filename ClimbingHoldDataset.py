from os import path, listdir
from numpy import loadtxt, array
from torch.utils.data import Dataset
from PIL import Image

class ClimbingHoldDataset(Dataset):
	def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None):
		# for annotations in annotations dir, each line is "label, YOLO format bounding box"
		# we only care about the label for classification
		rows = []
		for fname in listdir(annotations_dir):
			annotation_path = path.join(annotations_dir, fname)
			class_label = loadtxt(annotation_path, dtype=str)
			image_fname = path.splitext(fname)[0] + '.jpg'
			rows.append(array([image_fname, class_label]))

		self.img_labels = array(rows, dtype=str)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)
	
	def __getitem__(self, idx):
		image_fname, label = self.img_labels[idx]
		img_path = path.join(self.img_dir, image_fname)
		image = Image.open(img_path).convert("RGB")
		label = int(label)
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label
