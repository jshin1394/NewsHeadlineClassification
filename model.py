import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#TODO: To be further implemented with embedding if possible
class Simple_Net(nn.Module):
	def __init__(self):
		super(Simple_Net, self).__init__()
		self.fc1 = nn.Linear(10530, 5000)
		self.fc2 = nn.Linear(5000, 4)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x