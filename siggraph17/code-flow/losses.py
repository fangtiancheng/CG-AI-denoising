import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch

from torch.autograd import Variable
import torch.nn.functional as F


def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return func.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))

def edge_conv2d(im):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).type(torch.FloatTensor).to('cuda:0')

    edge_detect = conv_op(im)
    edge_detect = edge_detect.squeeze() # .detach().numpy()
    return edge_detect

def Sobel_loss(output, target):
	edge_detect_output = edge_conv2d(output)
	edge_detect_target = edge_conv2d(target)

	loss = torch.sum(torch.abs(edge_detect_target - edge_detect_output)) / torch.sum(edge_detect_target)

	return torch.pow(loss, 1.0 / 5)

def l1_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)

def get_temporal_data(output, target):
	final_output = output.clone()
	final_target = target.clone()
	final_output.fill_(0)
	final_target.fill_(0)
	# final_output = [torch.zeros()]
	# final_target = []

	for i in range(1, 7):
		# delta_output = output[:, i, :, :] - output[:, i-1, :, :]
		# delta_target = target[:, i, :, :] - target[:, i-1, :, :]
		# delta_output.unsqueeze_(1)
		# delta_target.unsqueeze_(1)
		# final_output.append(delta_output)
		# final_target.append(delta_target)
		final_output[:, i, :, :, :] = output[:, i, :, :] - output[:, i-1, :, :]
		final_target[:, i, :, :, :] = target[:, i, :, :] - target[:, i-1, :, :]
	# final_output = torch.cat(final_output, 1)
	# final_target = torch.cat(final_target, 1)
	# print(final_target.shape)
	return final_output, final_target

def temporal_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)

def loss_func(output, temporal_output, target, temporal_target):

	ls = l1_norm(output, target)
	# lg = HFEN(output, target)
	lg = Sobel_loss(output, target)
	lt = temporal_norm(temporal_output, temporal_target)

	# return 0.8 * ls + 0.15 * lg + 0.05 * lt, ls, lg, lt
	return 0.8 * ls + 0.15 * lg, ls, lg, 0
