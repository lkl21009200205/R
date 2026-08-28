
import torch 
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


# ==================== Lightweight CNN for FEMNIST ====================

class FEMNIST_CNN(nn.Module):
	"""Lightweight CNN designed for FEMNIST (1x28x28)."""

	def __init__(self, num_classes=62):
		super(FEMNIST_CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.pool = nn.MaxPool2d(2, 2)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B,32,14,14]
		x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B,64,7,7]
		x = self.dropout1(x)
		x = x.view(x.size(0), -1)  # [B,3136]
		x = F.relu(self.fc1(x))  # [B,128]
		x = self.dropout2(x)
		x = self.fc2(x)  # [B,62]
		return x

def get_model(name="vgg16", pretrained=True):
	if name == "FEMNIST_CNN":
		model = FEMNIST_CNN(num_classes=62)
	elif name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 
