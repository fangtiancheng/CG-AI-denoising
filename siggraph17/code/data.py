import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim
import re

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch


class RAEData(Dataset):
	def __init__(self, input_dir):
		super(RAEData, self).__init__()
		self.input_dir = input_dir
		def get_paths(root, dir):
			# get absolute path of image
			files = sorted(os.listdir(
				os.path.join(root, dir)))
			return [os.path.join(root, dir, file) for file in files] # 返回文件夹下的所有图片路径
		self.spp1 = get_paths(input_dir, 'out')
		self.ref = get_paths(input_dir, 'reference')
		# self.albedo = get_paths(input_dir, 'matrefl')
		self.albedo = get_paths(input_dir, 'albedo')
		self.normal = get_paths(input_dir, 'normal')
		self.depth = get_paths(input_dir, 'depth')
		# self.roughness = get_paths(input_dir, 'illum')
		self.roughness = get_paths(input_dir, 'roughness')
		# 获得文件路径，单一文件下的路径
		testimg = cv2.imread(self.spp1[0])[::2,::2,:] # 降采样，分辨率变为原来的1/4
		self.width = (testimg.shape[0]//32)*32 # 变成32的倍数
		self.height = (testimg.shape[1]//32)*32 # 变成32的倍数
		self.length = len(self.spp1)
		self._check()
	def _check(self):
		# check length
		assert(len(self.ref) == self.length)
		assert(len(self.albedo) == self.length)
		assert(len(self.normal) == self.length)
		assert(len(self.depth) == self.length)
		assert(len(self.roughness) == self.length)
		testimg = cv2.imread(self.spp1[0])
		assert(testimg.shape == cv2.imread(self.ref[0]).shape)
		assert (testimg.shape == cv2.imread(self.albedo[0]).shape)
		assert (testimg.shape == cv2.imread(self.normal[0]).shape)
		assert (testimg.shape == cv2.imread(self.depth[0]).shape)
		assert (testimg.shape == cv2.imread(self.roughness[0]).shape)
		print('shape = {}'.format(testimg.shape))
	def __getitem__(self, index):
		# 10spp_shading   ray_shading
		# 10spp_albedo    normal 
		# depth           roughness
		#-----------------------------

		# A = np.zeros((7, self.width, self.height, 8), dtype=np.float)
		# B = np.zeros((7, self.width, self.height, 3), dtype=np.float)
		# ALBEDO = np.zeros((7, self.width, self.height, 3), dtype=np.float)
		A, B, ALBEDO = [], [], []
		# batch size = 7
		for i in range(index*7, (index+1)*7):
			shading = cv2.imread(self.spp1[i])[::2,::2,:][:self.width, :self.height, :]
			ray_shading = cv2.imread(self.ref[i])[::2,::2,:][:self.width, :self.height, :]
			albedo = cv2.imread(self.albedo[i])[::2,::2,:][:self.width, :self.height, :]
			normal = cv2.imread(self.normal[i])[::2,::2,:][:self.width, :self.height, :]
			depth = cv2.imread(self.depth[i])[::2,::2,:][:self.width, :self.height, :]
			roughness = cv2.imread(self.roughness[i])[::2,::2,:][:self.width, :self.height, :]
			depth = (depth[:,:,0]+depth[:,:,1]+depth[:,:,2])/3
			roughness = (roughness[:,:,0]+roughness[:,:,1]+roughness[:,:,2])/3
			depth = np.expand_dims(depth, axis=2)
			roughness = np.expand_dims(roughness, axis=2)

			ray_shading = ray_shading.astype(np.float) / 255.0
			shading = shading.astype(np.float) / 255.0
			normal = normal.astype(np.float) / 255.0
			albedo = albedo.astype(np.float) / 255.0
			depth = depth.astype(np.float) / 255.0
			roughness = roughness.astype(np.float) / 255.0
			# print(depth.shape) # (576, 960, 3)
			# shading, normal 三维， 其余一维

			A_i = np.concatenate([shading, normal, depth, roughness], axis=2)
			A_i = np.expand_dims(A_i, axis=0)
			# print(A_i.shape) # (1, 576, 960, 8) 3 + 3 + 1 + 1
			A.append(A_i)
			B_i = np.expand_dims(ray_shading, axis=0)
			B.append(B_i)
			ALBEDO_i = np.expand_dims(albedo, axis=0)
			ALBEDO.append(ALBEDO_i)
			# A[i%7, :, :, :3] = shading  # 0 1 2
			# A[i%7, :, :, 3:6] = normal  # 3 4 5
			# A[i%7, :, :, 6:7] = depth   # 6
			# A[i%7, :, :, 7:8] = roughness  # 7

			# B[i%7, :, :, :] = ray_shading
			# ALBEDO[i%7, :, :, :] = albedo
		A = np.concatenate(A, axis=0)
		B = np.concatenate(B, axis=0)
		ALBEDO = np.concatenate(ALBEDO, axis=0)

		# print(A.shape, B.shape, ALBEDO.shape)
		# (7, 576, 960, 8) (7, 576, 960, 3) (7, 576, 960, 3)

		A = torch.from_numpy(A)
		B = torch.from_numpy(B)
		ALBEDO = torch.from_numpy(ALBEDO)
		# rgb通道转换
		A = A.permute((0, 3, 1, 2))
		B = B.permute((0, 3, 1, 2))
		ALBEDO = ALBEDO.permute((0, 3, 1, 2))

		return {
			'A': A.type(torch.float).to('cuda:0'),
			'B': B.type(torch.float).to('cuda:0'),
			'ALBEDO': ALBEDO.type(torch.float).to('cuda:0')
		} # A 通道，B参考，ALBEDO：ALBEDO

		
	def __len__(self):
		# reserve one batch for test
		return self.length//7 - 1

	def np_normalize(self, img):
		return (img - img.min()) / (img.max() - img.min())

	def save_image(self, img, img_name):
		img = torch.squeeze(img.detach(), dim=0) * 255.0
		img = img.permute((1, 2, 0))
		img = img.cpu().numpy().astype(np.uint8)
		
		cv2.imwrite(img_name, img)
