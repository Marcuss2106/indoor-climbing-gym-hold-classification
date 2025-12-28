import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn

from Net import Net
from ClimbingHoldDataset import ClimbingHoldDataset

NUM_CLASSES = 6

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
	torch.backends.cudnn.benchmark = True

def test_model(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Test", unit="batch") as pbar:
            for images, labels in dataloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)

                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
                pbar.update(1)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def label_to_onehot(y):
    # y is an int class index
    t = torch.zeros(NUM_CLASSES, dtype=torch.float)
    t.scatter_(dim=0, index=torch.tensor(y), value=1)
    return t

# mean = [0.6681055533042859, 0.6137803857704651, 0.5396168694315177]
# std  = [0.32921997147304766, 0.3079490945893265, 0.36234998143662933]
# mean rgb (170, 157, 138)
mean = [0.6681055533042859, 0.6137803857704651, 0.5396168694315177]
std  = [0.32921997147304766, 0.3079490945893265, 0.36234998143662933]

mean_t = torch.tensor(mean).view(3,1,1)
std_t  = torch.tensor(std).view(3,1,1)

def imshow(img):
	img = unnormalize(img)
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def unnormalize(img_tensor):
    # img_tensor: torch.Tensor [C,H,W], normalized
    return img_tensor * std_t + mean_t

def main():

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	
	print("Getting Testing Datasets...")

	test_dataset_path = Path('Processed_Dataset/test')
	test_dataset = ClimbingHoldDataset(
		annotations_dir=os.path.join(test_dataset_path, 'labels'),
		img_dir=os.path.join(test_dataset_path, 'images'),
		transform=transform,
		target_transform=label_to_onehot
	)

	classes = {
		0: 'Crimp',
		1: 'Jug',
		2: 'Pinch',
		3: 'Pocket',
		4: 'Slope',
		5: 'Volume',
	}

	print('------------')
	print(f"Testing dataset size: {len(test_dataset)} samples")
	print('------------')

	print("Getting Data Loaders..")

	test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False,
								pin_memory=True)

	net = Net().to(device)    
	print("Loading Model")
	checkpoint = torch.load("checkpoint.pth", map_location=device)
	net.load_state_dict(checkpoint["model_state"])

	print('------------')
	print(f'Testing Model')
	print('------------')
	# test_model(net, test_dataloader, device)
	test_dataiter = iter(test_dataloader)
	images, labels = next(test_dataiter)

	images_to_display = 5
	for j in range(images_to_display):
		for i in range(labels.shape[1]):
			if labels[j][i].item() == 1:
				print(f"{classes[i]}\t", end='')
	print()
	imshow(torchvision.utils.make_grid(images[:images_to_display]))

	outputs = net(images.to(device))
	_, predicted = torch.max(outputs, 1)
	predicted = predicted.cpu()
	labels = labels.cpu()
	probs = torch.softmax(outputs, dim=1).cpu()
	for j in range(min(images_to_display, labels.size(0))):
		actual_idx = labels.argmax(dim=1)[j].item()
		pred_idx = int(predicted[j].item())
		conf = probs[j, pred_idx].item()
		print(f"Actual: {classes[actual_idx]:10s}  Predicted: {classes[pred_idx]:10s}  Conf: {100*conf:.2f}%")
	
	print('------------')

	# prepare to count predictions for each class
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}

	with torch.no_grad():
		for images, labels in test_dataloader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			outputs = net(images)
			_, predictions = torch.max(outputs, 1)

			predictions = predictions.cpu()
			labels = labels.cpu()

			for label, pred in zip(labels, predictions):
				label = label.argmax().item()
				pred = pred.item()

				if pred == label:
					correct_pred[label] += 1
				total_pred[label] += 1


	# print accuracy for each class
	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		print(f'Accuracy for {classes[classname]}: {accuracy:.1f} %')

	print('------------')
	print('Finished Testing')
	print('------------')



if __name__ == "__main__":
	main()
