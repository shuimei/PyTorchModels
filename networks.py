from __future__ import print_function
import torch, torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

provided_models = {
	"resnet18": models.resnet18,
	"resnet34": models.resnet34,
	"resnet50": models.resnet50,
	"resnet101": models.resnet101,
	"resnet152":  models.resnet152,
	"vgg11": models.vgg11,
	"vgg11_bn": models.vgg11_bn,
	"vgg13": models.vgg13,
	"vgg13_bn": models.vgg13_bn,
	"vgg16": models.vgg16,
	"vgg16_bn": models.vgg16_bn,
	"vgg19": models.vgg19,
	"vgg19_bn":  models.vgg19_bn,
	"inception_v3": models.inception_v3,
	"alexnet" : models.alexnet,
	"squeezenet1_0" : models.squeezenet1_0,
	"squeezenet1_1" : models.squeezenet1_1,
	"densenet121" : models.densenet121,
	"densenet169" : models.densenet169,
	"densenet201" : models.densenet201,
	"densenet161" : models.densenet161,
	}


vgg16_pretrained = provided_models["vgg16"](pretrained=True)
