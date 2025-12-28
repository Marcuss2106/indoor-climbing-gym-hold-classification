from torch import nn

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear_relu_stack = nn.Sequential(
			nn.Conv2d(3, 16, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Flatten(),
			nn.Linear(32 * 32 * 32, 6),
			# nn.ReLU(),
			# nn.Linear(128, 6)
		)

	def forward(self, x):
		x = self.linear_relu_stack(x)
		return x
