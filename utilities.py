from __future__ import print_function
import torch.utils.data as Data
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os, torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_images_list(input_dir, label):
	labels_table = pd.read_csv(label)
	names = list(labels_table.name)
	labels = list(labels_table.invasive)
	images_list = list(map(lambda a: "%s/%s.jpg" % (input_dir, a), names))
	return images_list, labels
	
def img2tensor(input_img):
    arr = np.asarray(input_img.resize((224, 224)), dtype="float32")
    arr = arr.transpose()
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    img_tensor = torch.from_numpy(arr)
    return img_tensor


class ImageDataset(Data.Dataset):
	"""docstring for MyDataset"""

	def __init__(self, images, labels):
		super(ISMDataset, self).__init__()
		self.images = images
		self.labels = labels
		self.length = len(images)

	def __getitem__(self, index):
		img, target = self.images[index], self.labels[index]
		img_obj = Image.open(img).resize((224, 224))
		# img_obj = Image.open(img)
		arr = np.asarray(img_obj, dtype="float32")
		arr = arr.transpose()
		# arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
		img_tensor = torch.from_numpy(arr)
		target_tensor = torch.LongTensor(np.array([int(target)]))
		return img_tensor, target_tensor

	def __len__(self):
		return len(self.images)
