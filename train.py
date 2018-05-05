import sys
import re
import os.path
import unicodedata
import multiprocessing
import json

import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def train(model, train_iter, test_iter, args):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	steps = 0
	best_acc = 0
	train_acc = open("train_acc.txt", 'w')
	test_acc = open("test_acc.txt", "w")
	try:
		for epoch in range(1, args.epochs + 1):
			steps = 0
			for batch in train_iter:
				train_data, train_label = batch[0], batch[1]
				train_d = Variable(torch.from_numpy(train_data).type('torch.FloatTensor'))
				train_l = Variable(torch.from_numpy(train_label).type('torch.LongTensor'))
				optimizer.zero_grad()
				output = model(train_d)

				loss = F.cross_entropy(output, train_l, size_average=False)
				loss.backward()
				optimizer.step()

				steps += 1
				corrects = (torch.max(output, 1)[1].view(train_l.size()).data == train_l.data).sum()
				accuracy = 100.0 * corrects/len(train_l)
				if steps == 1:
					print(
							'\rEpoch[{}/{}], Step: [{}] - loss: {:.6f}, acc: {:.4f}%({}/{} )'.format(epoch, args.epochs, steps,
																						 loss.data[0], 
																						 accuracy,
																						 corrects,
																						 len(train_l)))	
					train_acc.write(str(accuracy) + '\n')
			eval_acc = eval(model, test_iter, args)
			test_acc.write(str(eval_acc) + '\n')
			if eval_acc > best_acc:
				best_acc = eval_acc
				save(model)
		train_acc.close()
		test_acc.close()

	except:
		print(steps)
		print(epoch)

def eval(model, eval_iter, args):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	steps = 0
	try:
		test_data, test_label = eval_iter[0], eval_iter[1]
		test_d = Variable(torch.from_numpy(test_data).type('torch.FloatTensor'))
		test_l = Variable(torch.from_numpy(test_label).type('torch.LongTensor'))
		optimizer.zero_grad()
		output = model(test_d)

		loss = F.cross_entropy(output, test_l, size_average=False)
		loss.backward()
		optimizer.step()

		steps += 1
		corrects = (torch.max(output, 1)[1].view(test_l.size()).data == test_l.data).sum()
		accuracy = 100.0 * corrects/len(test_l)
		if steps == 1:
			print(
					'\rTest Step: - loss: {:.6f}, acc: {:.4f}%({}/{} )'.format(
																				 loss.data[0], 
																				 accuracy,
																				 corrects,
																				 len(test_l)))
		return accuracy
	except:
		print(steps)

def test(model, test_iter, args):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	steps = 0
	try:
		test_data, test_label = test_iter[0], test_iter[1]
		test_d = Variable(torch.from_numpy(test_data).type('torch.FloatTensor'))
		test_l = Variable(torch.from_numpy(test_label).type('torch.LongTensor'))
		optimizer.zero_grad()
		output = model(test_d)

		loss = F.cross_entropy(output, test_l, size_average=False)
		loss.backward()
		optimizer.step()
		steps += 1
		corrects = (torch.max(output, 1)[1].view(test_l.size()).data == test_l.data).sum()
		accuracy = 100.0 * corrects/len(test_l)
		if steps == 1:
			print(
					'\rTest Step: - loss: {:.6f}, acc: {:.4f}%({}/{} )'.format(
																				 loss.data[0], 
																				 accuracy,
																				 corrects,
																				 len(test_l)))
	except:
		print(steps)

def save(model):
	save_path = 'model.pt'
	torch.save(model.state_dict(), save_path)