import pickle
import torch
import numpy as np
import random
from Datasets import *
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import torchvision

def loadpickle(fname):
	with open(fname, 'rb') as f:
		array = pickle.load(f, encoding='latin1')
	f.close()
	return array


class ImageDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.Resize((64, 64))])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		image = self.transforms(self.data[index])
		return image, self.labels[index]

def data_loader(dataset_path, pixel, batch_size):
	transforms = T.Compose([T.ToTensor(), T.Resize((64, 64)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_dataset = torchvision.datasets.CelebA('../../datasets', 'train', 'attr', transforms, download=False)
	test_dataset = torchvision.datasets.CelebA('../../datasets', 'test', 'attr', transforms, download=False)

	
	#indices_a = train_dataset.attr[:, 9] == 1 and train_dataset.attr[:, 20] == 0 # if you want to keep images with the label 5
	l1 = train_dataset.attr[:, 9] == 1
	l2 = train_dataset.attr[:, 20] == 0
	indices_a = []
	for i in range(0, len(l1)):
		indices_a.append(l1[i] and l2[i]) 
	indices_a = [i for i, x in enumerate(indices_a) if x]
	#indices_b = train_dataset.attr[:, 8] == 1 and train_dataset.attr[:, 20] == 0 # if you want to keep images with the label 5
	l1 = train_dataset.attr[:, 8] == 1
	l2 = train_dataset.attr[:, 20] == 0
	indices_b = []
	for i in range(0, len(l1)):
		indices_b.append(l1[i] and l2[i]) 
	indices_b = [i for i, x in enumerate(indices_b) if x]
	train_dataset = torch.utils.data.Subset(train_dataset, indices_a + indices_b)

	indices_a = test_dataset.attr[:, 9] == 1 # if you want to keep images with the label 5
	indices_a = [i for i, x in enumerate(indices_a) if x]
	indices_b = test_dataset.attr[:, 8] == 1 # if you want to keep images with the label 5
	indices_b = [i for i, x in enumerate(indices_b) if x]
	test_dataset = torch.utils.data.Subset(test_dataset, indices_a + indices_b)
	
	"""
	indices_a = train_dataset.attr[:, 20] == 1 # if you want to keep images with the label 5
	indices_a = [i for i, x in enumerate(indices_a) if x]
	indices_b = train_dataset.attr[:, 20] == 0 # if you want to keep images with the label 5
	indices_b = [i for i, x in enumerate(indices_b) if x]
	train_dataset = torch.utils.data.Subset(train_dataset, indices_a + indices_b)

	indices_a = test_dataset.attr[:, 20] == 1 # if you want to keep images with the label 5
	indices_a = [i for i, x in enumerate(indices_a) if x]
	indices_b = test_dataset.attr[:, 20] == 0 # if you want to keep images with the label 5
	indices_b = [i for i, x in enumerate(indices_b) if x]
	test_dataset = torch.utils.data.Subset(test_dataset, indices_a + indices_b)
	"""

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


	return train_loader, test_loader
