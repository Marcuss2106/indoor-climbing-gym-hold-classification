import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn

from Net import Net
from ClimbingHoldDataset import ClimbingHoldDataset

NUM_CLASSES = 6
PATIENCE = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
	torch.backends.cudnn.benchmark = True

def train_epoch(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    n_batches = len(dataloader)
    with tqdm(total=n_batches, desc="Train", unit="batch") as pbar:
        for i, (inputs, labels) in enumerate(dataloader, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
            pbar.update(1)

    return avg_loss

def eval_model(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Eval", unit="batch") as pbar:
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

def main():
	# mean = [0.6681055533042859, 0.6137803857704651, 0.5396168694315177]
	# std  = [0.32921997147304766, 0.3079490945893265, 0.36234998143662933]
	# mean rgb (170, 157, 138)
	mean = [0.6681055533042859, 0.6137803857704651, 0.5396168694315177]
	std  = [0.32921997147304766, 0.3079490945893265, 0.36234998143662933]

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	
	print("Getting Datasets...")
	train_dataset_path = Path('Processed_Dataset/train')
	train_dataset = ClimbingHoldDataset(
		annotations_dir=os.path.join(train_dataset_path, 'labels'),
		img_dir=os.path.join(train_dataset_path, 'images'),
		transform=transform,
		target_transform=label_to_onehot
	)

	validation_dataset_path = Path('Processed_Dataset/valid')
	validation_dataset = ClimbingHoldDataset(
		annotations_dir=os.path.join(validation_dataset_path, 'labels'),
		img_dir=os.path.join(validation_dataset_path, 'images'),
		transform=transform,
		target_transform=label_to_onehot
	)
	print('------------')
	print(f"Training dataset size: {len(train_dataset)} samples")
	print(f"Validation dataset size: {len(validation_dataset)} samples")
	print('------------')

	print("Getting Data Loaders..")
	train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True,
								pin_memory=True, persistent_workers=True, prefetch_factor=2)
	validation_dataloader = DataLoader(validation_dataset, batch_size=64, num_workers=4, shuffle=False,
								pin_memory=True, persistent_workers=True, prefetch_factor=2)

	# Counts for each of the classes in the training dataset
	counts = np.array([14689, 13848, 9482, 4274, 6599, 5605], dtype=np.float32)
	weights = 1.0 / counts
	weights = weights / weights.sum()
	class_weights = torch.tensor(weights, dtype=torch.float).to(device)

	net = Net().to(device)    
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
	start_epoch = 0
	if len(sys.argv) > 1 and sys.argv[1] == "load":
			print('------------')
			print("Loading Model")
			if len(sys.argv) > 2 and sys.argv[2]:
				checkpoint = torch.load(sys.argv[2], map_location=device)
				net.load_state_dict(checkpoint["model_state"])
				optimizer.load_state_dict(checkpoint["optimizer_state"])
				start_epoch = checkpoint["epoch"] + 1
			else:
				print("Give a valid checkpoint file!")
				print("Continuing with fresh network.")

	max_epochs = 25
	bad_epochs = 0
	best_acc = 0
	epoch_accuracy = []

	print('------------')
	if start_epoch != 0:
		print(f'Resuming from epoch {start_epoch}')
		print('------------')

	print('Baseline Evaluation')
	acc = eval_model(net, validation_dataloader, device)
	epoch_accuracy.append(acc)

	print('------------')
	print(f'Training for up to {max_epochs} epochs')
	print('------------')
	(Path("checkpoints/")).mkdir(parents=True, exist_ok=True)
	for epoch in range(max_epochs):
		print(f'Epoch {epoch + 1}/{max_epochs}')
		train_epoch(net, train_dataloader, criterion, optimizer, device)
		acc = eval_model(net, validation_dataloader, device)
		epoch_accuracy.append(acc)
		print("Saving Model...")
		torch.save({
			"epoch": epoch + start_epoch + 1,
			"model_state": net.state_dict(),
			"optimizer_state": optimizer.state_dict(),
		}, f"checkpoints/checkpoint_{epoch + 1}epoch.pth")
		if acc > best_acc:
			best_acc = acc
			bad_epochs = 0
		else:
			bad_epochs += 1
		if bad_epochs >= PATIENCE:
			print("Early Stop. No meaningful improvement.")
			break
		print('------------')
	print('Finished Training')
	print('------------')

	# Plot accuracy over epochs
	plt.plot(epoch_accuracy, marker='o')
	plt.title('Model Accuracy over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()

	print("Saving Model...")
	torch.save({
		"epoch": epoch + start_epoch,
		"model_state": net.state_dict(),
		"optimizer_state": optimizer.state_dict(),
	}, "checkpoint.pth")
	print("Done!")



if __name__ == "__main__":
	main()