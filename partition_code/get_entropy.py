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

def run(json_path, total_parts):
	dict_train={}
	part_no = 0
	
	while part_no < total_parts :
		g1 = dgl.distributed.load_partition(json_path, int(part_no))
		g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = g1
		labels = nfeat['_N/label']
	
		train_l = labels[nfeat['_N/train_mask'].bool()]
		uni, freq = th.unique(train_l, sorted=True, return_counts=True)
		uni, freq = list(map(int, uni.tolist())), freq.tolist()
		temp={}
		for key, value in zip(uni, freq):
			temp[int(key)] = value
		keys=part_no
		dict_train[keys]=temp
		part_no += 1
	  
	entropy=0
	dict_entropy={}
	total_train_nodes={}
	keys=0
	total_part_nodes=0
	for x in dict_train.values():
		entropy=0
		total_nodes=sum(x.values())
		total_part_nodes+=total_nodes
		total_train_nodes[keys]=total_nodes
		for k, i in x.items():
			if i!=0:
				p=i/total_nodes
				buffer=-p*np.log2(p)
				entropy=entropy+buffer
		dict_entropy[keys]=entropy
		keys+=1
	
	#print("Total Training nodes over partitions:  ",total_part_nodes)
	#print("Entropy of each partition: ")
	#print(dict_entropy)
	keys=0
	total_entropy=0
	for x in dict_train.values():
		#print(sum(x.values()))
		total_entropy+=(sum(x.values())/total_part_nodes)*dict_entropy[keys]
		keys+=1
	#print("Total Entropy: "+str(total_entropy))
	#
	#max_value=max(list(dict_entropy.values()))
	#min_value=min(list(dict_entropy.values()))
	#print("Entropy Gap between max and min value :       ",max_value-min_value)
	#print("Standard Deviation: ",statistics.stdev(list(dict_entropy.values())))
	return total_entropy, statistics.stdev(list(dict_entropy.values()))
