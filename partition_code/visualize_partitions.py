import pickle
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

ew ={0: {0: 8219, 1: 10, 2: 14, 3: 1, 4: 12, 5: 4, 6: 19, 7: 12, 8: 10, 9: 1431, 10: 4,11:0,12:0, 13: 32, 14: 2, 15: 14, 16: 26, 17: 1, 18: 8940, 19: 113, 20:0,21: 10, 22: 249, 23: 7723, 24: 2, 25: 56, 26: 2731, 27: 66, 28: 1, 29:31, 30: 63, 31: 3, 32: 18, 33: 6,34:0, 35: 11, 36: 19, 37: 17, 38: 7189, 39: 80, 40: 19}, 
     1: {0: 61, 1: 7, 2: 2184,3:0 ,4: 1528, 5: 7, 6: 10, 7: 48, 8: 10, 9: 7, 10: 3306, 11: 2, 12: 1129, 13: 1331, 14: 7, 15: 32, 16: 552, 17: 1, 18: 50, 19: 6771, 20: 6, 21: 2769, 22: 3501, 23: 164, 24: 23, 25: 992, 26: 90, 27: 3608,28:0, 29: 53, 30: 1508, 31: 9, 32: 830, 33: 36, 34: 3, 35: 2789, 36: 2795, 37: 2600, 38: 15, 39: 498, 40: 19},
     2: {0:121, 1: 2338, 2: 13, 3: 6, 4: 19, 5: 2370, 6: 2678, 7: 1336, 8: 7648, 9: 14, 10: 21, 11: 2006, 12: 1, 13: 268, 14: 3120, 15: 80, 16: 122, 17: 1640, 18: 87, 19: 125, 20: 1057, 21: 11, 22: 717, 23: 161, 24: 210,25: 45, 26: 44, 27: 233, 28: 15, 29: 3731, 30: 205, 31: 3205, 32: 78, 33: 3322, 34: 2270, 35: 24, 36: 29, 37: 148, 38: 45, 39: 268, 40: 21}, 
     3: {0: 56, 1: 5, 2: 15, 3: 10582, 4: 8, 5: 3, 6: 96, 7: 85, 8: 11, 9:3, 10: 5, 11: 1, 12: 1, 13: 250, 14: 6, 15: 16390, 16: 21, 17: 5, 18: 26, 19: 63, 20: 2, 21: 5, 22: 844, 23: 81, 24: 1, 25: 59, 26: 19, 27: 149, 28: 3257, 29: 36, 30: 99, 31: 7, 32: 170, 33: 12, 34: 3, 35: 17, 36: 12, 37: 61, 38: 31, 39: 1261, 40: 3312}}

eb ={0:{0: 2291, 1: 1372, 2: 421, 3: 1324, 4: 997, 5: 680, 6: 96, 7: 170, 8: 5508, 9: 131, 10: 3249, 11: 5, 12: 1101, 13: 333, 14: 788, 15: 3310, 16: 155, 17: 5, 18: 63, 19: 5795, 20: 8, 21: 24, 22: 1822, 23: 3730, 24: 132, 25: 67, 26: 84, 27: 762, 28: 49, 29: 1422, 30: 608, 31: 26, 32: 456, 33: 605, 34: 852, 35: 5, 36: 529, 37: 546, 38: 119, 39: 568, 40: 177},
       1:{0: 2730, 1: 10, 2: 22, 3: 4948, 4: 15, 5: 1318, 6: 2545, 7: 63, 8: 87, 9: 21, 10: 14, 11: 1999, 12: 4, 13: 316, 14: 745, 15: 156, 16: 123, 17: 425, 18: 6672, 19: 286, 20: 1, 21: 4, 22: 1287, 23: 420, 24: 66, 25: 108, 26: 104, 27: 215, 28: 150, 29: 1827, 30: 279, 31: 54, 32: 88, 33: 141, 34: 411, 35: 15, 36: 375, 37: 1903, 38: 5279, 39: 946, 40: 58},
       2:{0: 339, 1: 4, 2: 77, 3: 2432, 4: 511, 5: 52, 6: 41, 7: 1204, 8: 801, 9: 1279, 10: 23, 11: 2, 12: 24, 13: 873, 14: 49, 15: 9348, 16: 146, 17: 7, 18: 152, 19: 719, 20: 1053, 21: 2745, 22: 1381, 23: 1779, 24: 16, 25: 143, 26: 2622, 27: 732, 28: 21, 29: 94, 30: 505, 31: 1069, 32: 420, 33: 183, 34: 960, 35: 573, 36: 1893, 37: 146, 38: 1720, 39: 249, 40: 1341},
       3:{0: 3097, 1: 974, 2: 1706, 3: 1885, 4: 44, 5: 334, 6: 121, 7: 44, 8: 1283, 9: 24, 10: 50, 11: 3, 12: 2, 13: 359, 14: 1553, 15: 3702, 16: 297, 17: 1210, 18: 2216, 19: 272, 20: 3, 21: 22, 22: 821, 23: 2200, 24: 22, 25: 834, 26: 74, 27: 2347, 28: 3053, 29: 508, 30: 483, 31: 2075, 32: 132, 33: 2447, 34: 53, 35: 2248, 36: 58, 37: 231, 38: 162, 39: 344, 40: 1795}
       }

dictionary = ew  # Your label frequencies dictionary

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
plt.subplot(1,2,1)
cmap = plt.get_cmap('rainbow')

# Create bar plot
classes = range(41)
color = ['r','b','g','tab:orange']
btm = np.array([0 for i in range(41)])
for i in range(4):
    partition_frequencies = [label_frequencies[c][i] for c in range(41)]
    bars = plt.bar(classes, partition_frequencies, bottom = btm, color=color[i], edgecolor='black')
    btm += np.array(partition_frequencies) 

# Create legend
legend_labels = [f'Partition {i}' for i in range(len(dictionary))]
plt.legend(legend_labels, fontsize=14)
plt.xlabel('Labels\n(a)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(range(0,41,5),fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig("ew.pdf", format="pdf", bbox_inches="tight")

dictionary = eb  # Your label frequencies dictionary

# Calculate total frequencies for each label in each partition
label_frequencies = {}
for partition, inner_dict in dictionary.items():
    for label, frequency in inner_dict.items():
        if label in label_frequencies:
            label_frequencies[label][partition] = frequency
        else:
            label_frequencies[label] = {partition: frequency}

# Adjust the figure size
plt.subplot(1,2,2)

# Create color map
cmap = plt.get_cmap('rainbow')

# Create bar plot
classes = range(41)
color = ['r','b','g','tab:orange']
btm = np.array([0 for i in range(41)])
for i in range(4):
    partition_frequencies = [label_frequencies[c][i] for c in range(41)]
    bars = plt.bar(classes, partition_frequencies, bottom = btm, color=color[i], edgecolor='black')
    btm += np.array(partition_frequencies) 

# Create legend
legend_labels = [f'Partition {i}' for i in range(len(dictionary))]
plt.legend(legend_labels, fontsize=14)
plt.xlabel('Labels\n(b)', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
plt.xticks(range(0,41,5), fontsize=14)
plt.yticks([])
plt.savefig("combined2.pdf", format="pdf", bbox_inches="tight")