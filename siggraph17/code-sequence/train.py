from typing import Sequence
import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim
import time

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch, argparse, pdb
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from tensorboardX import SummaryWriter


from model import *
from data import *
from losses import *
# multiprocessing.set_start_method('spawn')

def save_checkpoint(state, filename):
	torch.save(state, filename)

def train_sequence(model, sequence, sequence_sum):
	# output_final = sequence['B'].clone()
	# output_final.fill_(0)
	# target_final = sequence['B'].clone()
	# target_final.fill_(0)
	output_final = []
	target_final = []

	inp = sequence['A']
	target = sequence['B']

	loss_final = 0
	ls_final = 0
	lg_final = 0
	lt_final = 0

	for j in range(0, sequence_sum):
		inpi = inp[:, j, :, :, :]
		gti = target[:, j, :, :, :]

		final_inp = {
			'A': inpi,
			'B': gti
		}

		# print(inpi.shape) # torch.Size([1, 8, 576, 960])

		model.set_input(final_inp)
		# if j == 0:
		model.reset_hidden()

		output = model().unsqueeze_(1)
		output_final.append(output)
		gti.unsqueeze_(1)
		output_final.append(output)
		target_final.append(gti)
		# output_final[:, j, :, :, :] = output
		# target_final[:, j, :, :, :] = gti
	output_final = torch.cat(output_final, 1)
	target_final = torch.cat(target_final, 1)
	# temporal_output, temporal_target = get_temporal_data(output_final, target_final) # 时序信息的差值

	for j in range(0, sequence_sum):
		output = output_final[:, j, :, :, :]
		target = target_final[:, j, :, :, :]
		# t_output = temporal_output[:, j, :, :, :]
		# t_target = temporal_target[:, j, :, :, :]
		t_output, t_target = None, None

		l, ls, lg, lt = loss_func(output, t_output, target, t_target)
		loss_final += l
		ls_final += ls
		lg_final += lg
		lt_final += lt

	return loss_final, ls_final, lg_final, lt_final


def train(model, dataset, optimizer, epoch, now, sequence_sum):

	total_loss = 0
	total_loss_num = 0

	for i, item in enumerate(dataset):
		optimizer.zero_grad()
		loss_final, ls_final, lg_final, lt_final = train_sequence(model, item, sequence_sum)
		
		loss_final.backward(retain_graph=False)
		optimizer.step()

		total_loss += loss_final.item()
		total_loss_num += 1

		niter = epoch * len(dataset) + i
		writer.add_scalars('Train_loss' + now, {"Train_loss": loss_final.item()}, niter)

		if i % 50 == 0:
			# print('[Epoch : %s] [%s/%s] Loss => %s , L1 => %s , HFEN => %s , TEMPORAL => %s' %
			# 		(epoch+1, (i+1), len(data_loader), loss_final.item(), ls_final.item(),
			# 			lg_final.item(), lt_final.item()))
			print('[Epoch : %s] [%s/%s] Loss => %s , L1 => %s , HFEN => %s' %
					(epoch+1, (i+1), len(data_loader), loss_final.item(), ls_final.item(),
						lg_final.item()))
			sys.stdout.flush()

	total_loss /= total_loss_num

	return total_loss



if __name__ == '__main__':

	# added when there are more than 1 GPUs
	torch.multiprocessing.set_start_method('spawn')
	writer = SummaryWriter('./logs')

	now = time.asctime().replace(" ", "_")

	parser = argparse.ArgumentParser(description='RecurentAE, SIGGRAPH \'17')
	parser.add_argument('--data_dir', type=str, help='Data directory')
	parser.add_argument('--save_dir', type=str, help='Model chekpoint saving directory')
	parser.add_argument('--name', type=str, help='Experiment Name')
	parser.add_argument('--epochs', type=int, help='Number of epochs to train')
	parser.add_argument('--sequence', type=int, help='Number of sequence', default=7)
	# parser.add_argument('--load', type=bool, default=False, help='load latest checkpoint from save_dir')
	parser.add_argument('--load', type=str, default="", help='load latest checkpoint from save_dir')
	args = parser.parse_args()

	sequence_sum = args.sequence
	print("The sequence num now is : ", sequence_sum, " ... ")
	data_loader = RAEData(args.data_dir, sequence_sum)
	dataset = DataLoader(data_loader, batch_size=2, num_workers=2, shuffle=True)

	if args.load != "":
		chkpoint = torch.load(args.load)
		model = RecurrentAE(8)
		model.to('cuda:0')
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
		# lr scheduler
		scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
		print("Model loaded! ...")
		epoch_ = chkpoint['epoch']
		model.load_state_dict(chkpoint['state_dict'])
		optimizer.load_state_dict(chkpoint['optimizer'])
		scheduler.load_state_dict(chkpoint['scheduler'])
	else:
		model = RecurrentAE(8)
		model.to('cuda:0')

		epoch_ = 0
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
		scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)

	for epoch in range(epoch_, args.epochs):
		
		print('\nEpoch %s' % (epoch+1))

		total_loss = train(model, dataset, optimizer, epoch, now, sequence_sum)

		print('Epoch %s loss => %s' % (epoch+1, total_loss))
		sys.stdout.flush()
		scheduler.step()

		if (epoch + 1) % 20 == 0:
			print('SAVING MODEL AT EPOCH %s' % (epoch+1))
			save_checkpoint({
					'epoch': epoch+1,
					'state_dict':model.state_dict(),
					'optimizer':optimizer.state_dict(),
					'scheduler':scheduler.state_dict(),
				}, '%s/%s_%s.pt' % (args.save_dir, args.name, epoch+1))


	# save_checkpoint({
	# 			'epoch': args.epochs,
	# 			'state_dict':model.state_dict(),
	# 			'optimizer':optimizer.state_dict(),
	# 		}, '%s/%s_%s.pt' % (args.save_dir, args.name, args.epochs))
