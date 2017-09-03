from __future__ import print_function
import torch, torchvision
import torch.autograd.variable as Variable
import torchvision.models as models
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def image_label(image_dir, label_file, image_format="png"):
	'''
	image_label: read a csv file that contains image names and corresbonding labels.
	params:
		(string) image_dir: a directory where images are stored
		(string) label_file: a csv file with fixed header(name, label). name is for image name, and label is for the image lable
		(string) image_format: image's format, say "png" "jpg" etc.
	return:
		(tuple) a tuple of two lists, images name list and label list
	'''	
	labels_table = pd.read_csv(label_file)
	names = list(labels_table.name)
	labels = list(labels_table.label)
	images_list = list(map(lambda n: "%s/%s" % (image_dir, n), names))
	return images_list, labels


def image_to_tensor(input_image, format="png", size=(224, 224)):
	'''
	image_to_tensor: convert an image from an image object to a tensor
	params:
		(PIL.Image) input_image: an PIL.Image object
		(string) size: (optional) to adopt some fixed models, size has to be fixed, too.
	return:
		(torch.tensor) a torch.tensor object
	'''

	arr = np.asarray(input_image.resize(size), dtype="float32")
	if format == "png":
		arr = arr[:,:,:3]
	else:
		pass
	arr = arr.transpose()
	arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
	img_tensor = torch.from_numpy(arr)
	return img_tensor

class ImageDataset(Data.Dataset):
	'''an Image Dataset constructor
	derived from torch.utils.data.Dataset, this class is used to make pairs of images and 
	labels to be visited like a list.
	params:
		(list) images: a image names list, with file path
		(list) labels: a label list
		(tuple) image_size: optional image size if resize is required
	attributes:
		(list) __images: same as input image names list
		(list) __labels: same as input label list
		(int) __length: the length of the dataset
	methods:
		(list) get_images_list: private variable visit method
		(list) get_labels_list: private variable visit method
		(int) length: private variable visit method
		(int) __len__: a magic method to get length
		(torch.tensor) image_to_tensor: private method, convert a PIL.Image object into
		torch.tensor object
		(tuple) __getitem__: a magic method implemented to visit data like slice
	'''

	def __init__(self, images, labels, image_size=(224, 224)):
		super(ImageDataset, self).__init__()
		try:
			if len(images) == len(labels):
				self.__images = images
				self.__labels = labels
				self.__image_size = image_size
				self.__length = len(images)
		except Exception as e:
			print("Length Unmatching: length of images not equal to length of labels")
			raise e

	def get_images_list(self):
		return self.__images

	def get_lablels_list(self):
		return self.__labels

	def length(self):
		return self.__length

	def __len__(self):
		return self.__length
	
	def image_to_tensor(self, input_image, format="png"):
		'''
		image_to_tensor: convert an image from an image object to a tensor
		params:
			(PIL.Image) input_image: an PIL.Image object
			(string) size: (optional) to adopt some fixed models, size has to be fixed, too.
		return:
			(torch.tensor) a torch.tensor object
		'''
		arr = np.asarray(input_image.resize(self.__image_size), dtype="float32")
		if format == "png":
			arr = arr[:,:,:3]
		arr = arr.transpose()
		arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
		image_tensor = torch.from_numpy(arr)
		return image_tensor
	
	def __getitem__(self, index):
		image, target = Image.open(self.__images[index]), self.__labels[index]
		image_tensor = self.image_to_tensor(image)
		target_tensor = torch.LongTensor(np.array([int(target)]))
		return image_tensor, target_tensor

	

