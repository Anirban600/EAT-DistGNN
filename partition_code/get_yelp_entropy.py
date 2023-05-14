import dgl
import torch as th
import numpy as np
import sys
from collections import Counter
from ogb.nodeproppred import DglNodePropPredDataset
import matplotlib.pyplot as plt
import pandas as pd
stdoutOrigin = sys.stdout
import copy
from math import log2
import statistics
import argparse


def get_d(mask,label):
	num_zeros_dict = {}
	num_ones_dict = {}
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",num_parts)
	t_label=th.index_select(label,0,th.nonzero(mask, as_tuple=True)[0])
        #train_l = th.index_select(label,0,th.nonzero(train_l, as_tuple=True)[0])
        #print(t_label.shape[0])
        #print(t_label.shape[1])
	for dim in range(t_label.shape[1]):
		num_zeros = th.sum(t_label[:, dim] == 0)
                #print("0 s:    ",num_zeros)
		num_ones = th.sum(t_label[:, dim] == 1)
                #print("1 s:    ",num_ones)
		num_zeros_dict[dim] = num_zeros.item()
		num_ones_dict[dim] = num_ones.item()
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(num_zeros_dict)
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(num_ones_dict)
        #print("Done..", num_parts)
        #print("******************************************************************")
	return [num_zeros_dict,num_ones_dict,t_label.shape[0]]

# Entropy of a given label distribution x
def ent(_zeros_dict,_ones_dict):
	sum_dict = {}
	for key in _zeros_dict.keys():
		sum_dict[key] = _zeros_dict[key] + _ones_dict[key]
	p = {}
	q = {}
	plog2p_dict = {}
	qlog2q_dict = {}
	entro={}
	for key in _zeros_dict.keys():
		p = _zeros_dict[key] / sum_dict[key]
		q = _ones_dict[key] / sum_dict[key]
		if p == 0:
			plog2p_dict[key] = 0
		else:
			plog2p_dict[key] = p * log2(p)
		if q == 0:
			qlog2q_dict[key] = 0
		else:
			qlog2q_dict[key] = q * log2(q)
		entro[key] = -plog2p_dict[key] - qlog2q_dict[key]
	total_sum = sum(entro.values())
        #print(total_sum)
	return total_sum


def run(json_path, total_parts):
	def get_d(mask,label):
		num_zeros_dict = {}
		num_ones_dict = {}
        	#print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",num_parts)
		t_label=th.index_select(label,0,th.nonzero(mask, as_tuple=True)[0])
        	#print(t_label.shape[1])
		for dim in range(t_label.shape[1]):
			num_zeros = th.sum(t_label[:, dim] == 0)
                	#print("0 s:    ",num_zeros)
			num_ones = th.sum(t_label[:, dim] == 1)
                	#print("1 s:    ",num_ones)
			num_zeros_dict[dim] = num_zeros.item()
			num_ones_dict[dim] = num_ones.item()
		return [num_zeros_dict,num_ones_dict,t_label.shape[0]]
	# Entropy of a given label distribution x
	def ent(_zeros_dict,_ones_dict):
		sum_dict = {}
		for key in _zeros_dict.keys():
			sum_dict[key] = _zeros_dict[key] + _ones_dict[key]
		p = {}
		q = {}
		plog2p_dict = {}
		qlog2q_dict = {}
		entro={}
		for key in _zeros_dict.keys():
			p = _zeros_dict[key] / sum_dict[key]
			q = _ones_dict[key] / sum_dict[key]
			if p == 0:
				plog2p_dict[key] = 0
			else:
				plog2p_dict[key] = p * log2(p)
			if q == 0:
				qlog2q_dict[key] = 0
			else:
				qlog2q_dict[key] = q * log2(q)
			entro[key] = -plog2p_dict[key] - qlog2q_dict[key]
		total_sum = sum(entro.values())
        	#print(total_sum)
		return total_sum


	dict_train={}
	dict_total={}
	part_no = 0
	
	while part_no < total_parts :
		g1 = dgl.distributed.load_partition(json_path, int(part_no))
		g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = g1
		labels = nfeat['_N/label']

		zeros_dict,ones_dict,total_nodes=get_d(nfeat['_N/train_mask'] ,labels)
		dict_entropy_train=ent(zeros_dict,ones_dict)
                #print(dict_entropy_train)
		dict_train[part_no]=dict_entropy_train
		dict_total[part_no]=total_nodes
                #print(">>>>>>>>>> For Test >>>>>>>>>>>>>")
		zeros_dict,ones_dict,tot=get_d(nfeat['_N/test_mask'] ,labels)
		dict_entropy_test=ent(zeros_dict,ones_dict)
		part_no+=1
	total_sum=sum(dict_total.values())
	total_entropy=0
	for x in dict_train.keys():
		total_entropy+=( dict_total[x]/total_sum)*dict_train[x]
	return total_entropy, statistics.stdev(list(dict_train.values()))
