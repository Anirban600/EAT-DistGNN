import pickle
import numpy as np
import pandas as pd
import torch as th
import dgl

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def visualize_partitions(json_path, total_parts):
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

    dictionary = dict_train  # Your label frequencies dictionary
    num_classes = len(dictionary[0].keys())

    # Calculate total frequencies for each label in each partition
    label_frequencies = {}
    for partition, inner_dict in dictionary.items():
        for label, frequency in inner_dict.items():
            if label in label_frequencies:
                label_frequencies[label][partition] = frequency
            else:
                label_frequencies[label] = {partition: frequency}
    # Adjust the figure size
    plt.figure(figsize=(10, 4))

    # Create color map
    # plt.subplot(1,2,1)
    cmap = plt.get_cmap('rainbow')

    # Create bar plot
    classes = range(num_classes)
    color = ['r','b','g','tab:orange']
    btm = np.array([0 for i in range(num_classes)])
    for i in range(total_parts):
        partition_frequencies = [label_frequencies[c][i] if i in label_frequencies[c].keys() else 0 for c in range(num_classes)]
        bars = plt.bar(classes, partition_frequencies, bottom = btm, color=color[i], edgecolor='black')
        btm += np.array(partition_frequencies) 

    # Create legend
    legend_labels = [f'Partition {i}' for i in range(len(dictionary))]
    path_components = json_path.split("/")
    save_path = "/".join(path_components[:-2])+f"/{path_components[-2]}"
    plt.legend(legend_labels, fontsize=14)
    plt.xlabel('Labels\n(a)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(range(0,num_classes,1),fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{save_path}_distribution.jpg", format="jpg", bbox_inches="tight")