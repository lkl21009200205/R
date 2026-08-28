

import torch 
from torchvision import datasets, transforms

def get_dataset(dir, name):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		# Data transforms for the training dataset.
		transform_train = transforms.Compose([
			# Randomly crop images so the model sees different regions during training, improving generalization.
			transforms.RandomCrop(32, padding=4),
			# Randomly flip images horizontally with a 50% probability.
			transforms.RandomHorizontalFlip(),
			# Convert images to PyTorch tensor format.
			transforms.ToTensor(),
			# Normalize images.
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		# Data transforms for the test dataset.
		transform_test = transforms.Compose([
			# Same as the last two transforms used for the training dataset.
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		# transform=transform_train applies the previously defined crop, flip, tensor conversion, and normalization steps to the loaded training data.
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										 transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

	elif name == 'cifar100':
		# Common CIFAR-100 mean and standard deviation values precomputed on the training set.
		mean = (0.5071, 0.4865, 0.4409)
		std = (0.2673, 0.2564, 0.2762)

		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		])
		train_dataset = datasets.CIFAR100(dir, train=True, download=True,
										  transform=transform_train)
		eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
	
	return train_dataset, eval_dataset
