

import torch 
from torchvision import datasets, transforms

def get_dataset(dir, name):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		# 对训练数据集的数据的操作
		transform_train = transforms.Compose([
			# 图像进行随机裁剪，让模型在训练过程中看到不同区域的图像，提高模型的泛化能力
			transforms.RandomCrop(32, padding=4),
			# 随机地将图像进行水平翻转，概率为50%
			transforms.RandomHorizontalFlip(),
			# 将图像转换为PyTorch的张量格式
			transforms.ToTensor(),
			# 对图像进行标准化
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		# 对测试数据集的数据的操作
		transform_test = transforms.Compose([
			# 与训练数据集的最后两个操作一样
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		# transform=transform_train：将之前定义的 transform_train 应用到加载的训练数据上，即对图像进行裁剪、翻转、转换为张量以及标准化。
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										 transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

	elif name == 'cifar100':
		# CIFAR-100 常用的均值和标准差（预先在训练集上计算得到）
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