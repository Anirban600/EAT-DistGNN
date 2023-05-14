import argparse
import os
import sys
import time

import numpy as np
import torch as th

import dgl
from collections import Counter
import math

import copy

import random

from math import log2
import dgl
import dump_part
from dgl.data import YelpDataset
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from load_graph import load_ogb, load_reddit
#from load_graph_custom import load_yelp

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogb-product, ogb-paper100M,flicker",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument('--sample_length', type=int, default='5',
                           help='length of sample node.')
    argparser.add_argument('--reshuffle', type=bool,
                           help='reshuffle is allowed or not')
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--grp_parts", type=int, default=4, help="group of partitions"
    )
    argparser.add_argument(
        "--num_runs", type=int, default=15, help="number of runs"
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "reddit":
        dataset = RedditDataset()
        g = dataset[0]
        #g, _ = load_reddit()
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
        g = dataset[0]
        #g, _ = load_ogb("ogbn-products")
    elif args.dataset == "ogbn-papers100M":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M'))
        g = dataset[0]
        #g, _ = load_ogb("ogbn-papers100M")
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv'))
        g = dataset[0]
        #g, _ = load_ogb("ogbn-arxiv")
    elif args.dataset == "flickr":
        dataset = FlickrDataset()
        g = dataset[0]
        #g, _ = load_flicker()
    elif args.dataset == "yelp":
        dataset = YelpDataset()
        g = dataset[0]
        #g, _ = load_yelp()
    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.number_of_nodes(), g.number_of_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    #print(g.ndata["label"].numpy())
    #print("CLASSES:      ",np.unique(g.ndata["label"].numpy()))
    #print("NUMBER:      ",len(np.unique(g.ndata["label"].numpy())))
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
   
    
    #cosi = th.nn.CosineSimilarity(dim=1)
    #cosir = th.nn.CosineSimilarity(dim=0)
    runstart = time.time()
   
    labels_copy = copy.deepcopy(g.ndata["label"])
    #print(labels_copy)
    #features=copy.deepcopy(g.ndata["features"])
  
# Metis Call
    print("...........................Metis start.............................")
    start = time.time()
    membership=dgl.metis_partition_assignment(g, args.num_parts, balance_ntypes=balance_ntypes, balance_edges=args.balance_edges, mode='k-way', objtype='cut')
    print("Total Metis Time : ", time.time()-start)
    #print(len(membership))
    print("............................Metis end..............................")
    
    def get_d(num_parts,membership,mask,label):
        num_zeros_dict = {}
        num_ones_dict = {}
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",num_parts)
        t_label=th.index_select(label,0,th.nonzero(th.where(membership==num_parts,1,0)*mask, as_tuple=True)[0])
        #print(t_label.shape[0])
        for dim in range(t_label.shape[1]):
                num_zeros = th.sum(t_label[:, dim] == 0)
                #print("0 s:    ",num_zeros)
                num_ones = th.sum(t_label[:, dim] == 1)
                #print("1 s:    ",num_ones)
                num_zeros_dict[dim] = num_zeros.item()
                num_ones_dict[dim] = num_ones.item()
        return [num_zeros_dict,num_ones_dict]
    
     
    def get(num_parts,membership,mask,label):
        zeros_dict = {}
        ones_dict = {}
        for part in range(num_parts):
            z,o=get_d(part,membership,mask,label)
            zeros_dict[part] = z
            ones_dict[part] = o
        return [zeros_dict,ones_dict]
    
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
                plog2p_dict[key] = p * math.log2(p)
            if q == 0:
                qlog2q_dict[key] = 0
            else:
                qlog2q_dict[key] = q * math.log2(q)
            entro[key] = -plog2p_dict[key] - qlog2q_dict[key]
        total_sum = sum(entro.values())
        #print(total_sum)
        return total_sum
    
    # Entropy of all partitions
    def entropy(zeros_dict_,ones_dict_):
        dict_entropy={}
        total_entropy=0
        total_nodes=0
        for key in zeros_dict_.keys():
            dict_entropy[key]=ent(zeros_dict_[key],ones_dict_[key])
            total_nodes+=zeros_dict_[key][0]+ones_dict_[key][0]
        #print(total_nodes)
        #for key in dict_entropy.keys():
        total_entropy=0
        for key in zeros_dict_.keys():
            #print(zeros_dict_[key][0]+ones_dict_[key][0])
            total_entropy+=((zeros_dict_[key][0]+ones_dict_[key][0])/total_nodes)*dict_entropy[key]
        #print("Total Entropy: "+str(total_entropy))
        return [dict_entropy,total_entropy]
          
        
    #print("....................dict train.........................")
    zeros_dict,ones_dict=get(args.num_parts,membership,g.ndata["train_mask"],labels_copy)
    temp_=Counter(zeros_dict[0])+Counter(ones_dict[0])
    dict_entropy,pre_total_entropy=entropy(zeros_dict,ones_dict)
    
    #get minimum value
    min_total_entropy_list=[]
    min_total_entropy=100
    min_new_membership=th.zeros(len(membership))
    min_run=0
    begin_start=time.time()
    for repeat in range(args.num_runs):
        print("Run : ",repeat)
        loopstart=time.time()

    
        # initialise the compute nodes 
        keys_val = list(dict_entropy.keys())
        #print(keys_val)
        random.shuffle(keys_val)
        #keys_val = [7, 49, 14, 80, 47, 85, 2, 36, 67, 70, 45, 94, 75, 76, 90, 34, 40, 81, 51, 77, 54, 24, 64, 20, 18, 58, 92, 82, 35, 41, 32, 15, 13, 38, 87, 44, 71, 86, 37, 12, 23, 6, 84, 56, 27, 60, 88, 16, 0, 97, 63, 11, 8, 29, 69, 42, 9, 96, 65, 46, 31, 78, 28, 99, 33, 68, 59, 53, 19, 22, 61, 26, 62, 3, 55, 79, 95, 91, 72, 52, 25, 5, 17, 30, 48, 4, 66, 1, 50, 21, 10, 73, 43, 57, 74, 89, 98, 93, 83, 39]
        print("Sequence :", keys_val)
        Finalpart={}
        load_partition={}
        Final_part_zero={}
        Final_part_ones={}
        #Final_part_trainfeat={}
        #Final_part_testfeat={}
        #Final_part_total={}
    
        for i in range(args.grp_parts):
            load=np.zeros(int(args.num_parts/args.grp_parts))
            load[0]=keys_val[i]
            Final_part_zero[i]=zeros_dict[keys_val[i]]
            Final_part_ones[i]=ones_dict[keys_val[i]]
            #Final_part_total[i]=total_samples[i]
            #Final_part_trainfeat[i]=dict_train_feat[keys_val[i]]
            #Final_part_testfeat[i]=dict_test_feat[keys_val[i]]
            Finalpart[i]=load
            load_partition[i]=int(args.num_parts/args.grp_parts)-1
    
        #print(Finalpart)
        #print(Final_part_zero)
        #print(load_partition)
        #print(Final_part_ones)
        
        
        ##########################################################
        
        # cluster remaining partitions to compute nodes
        key_val=keys_val[args.grp_parts:]
        l=list(range(args.grp_parts))
        for k in range(len(key_val)):
            min_inx=0
            min_diff=100
            for x in key_val:
                #print("Check for parttion:   "+str(x))
                for i in l:
                    temp_zeros=Counter(Final_part_zero[i])+Counter(zeros_dict[x])
                    temp_ones=Counter(Final_part_ones[i])+Counter(ones_dict[x])
                    entr_i_and_x = ent(dict(temp_zeros),dict(temp_ones))
                    entr_i=ent(Final_part_zero[i],Final_part_ones[i])
                    entr_x=ent(zeros_dict[x],ones_dict[x])
                    diff_ent=entr_i_and_x - entr_i - entr_x
                    #temp_trainfeat=Final_part_trainfeat[i]+dict_train_feat[x]
                    #temp_testfeat=Final_part_testfeat[i]+dict_test_feat[x]
                    #cos_sim=cosir(temp_trainfeat,temp_testfeat)
                    diff=diff_ent
                    #diff=diff_ent-4*cos_sim
                    #print("With group "+str(i)+ "  Entropy difference:  "+str(diff_ent)+"Total diff"+str(diff))
                    #print("With group "+str(i)+" Feature similarity "+str(cos_sim)+"Entropy difference:  "+str(diff_ent)+"Total diff"+str(diff))
                    if min_diff>diff:
                        min_diff=diff
                        min_inx=i
                        keep_x=x
            Finalpart[min_inx][load_partition[min_inx]]=keep_x
            load_partition[min_inx]-=1
            #Finalpart[min_inx][]
            if load_partition[min_inx]==0:
                l.remove(min_inx)
            #if len(l)==0:
            #    l=list(range(args.grp_parts))
            #print(l)
            #print(load_partition)
            #------------------------------------------
            #l.remove(min_inx)
            key_val.remove(keep_x)
            #if len(l)==0:
            #    l=list(range(args.grp_parts))
            #print(l)
            #print(Finalpart)
            #print("parttion "+str(keep_x)+" is assigned to group "+str(min_inx))
            #Finalpart[min_inx]=Finalpart[min_inx]+part_mask[keep_x]
            Final_part_zero[min_inx]=dict(Counter(Final_part_zero[min_inx])+Counter(zeros_dict[keep_x]))
            Final_part_ones[min_inx]=dict(Counter(Final_part_ones[min_inx])+Counter(ones_dict[keep_x]))
            #Final_part_trainfeat[min_inx]=Final_part_trainfeat[min_inx]+dict_train_feat[keep_x]
            #Final_part_testfeat[min_inx]=Final_part_testfeat[min_inx]+dict_test_feat[keep_x]
            
            #print("Final Assignment: ",load_partition)
            #print(Final_part_trainfeat)
            
        new_membership=th.zeros(len(membership))
    # Membership of compute nodes
        for i in range(args.grp_parts):
            if i !=0:
                new_mem=th.zeros(len(membership))
                for j in list(Finalpart[i]):
                    new_mem+=th.where(membership==j,1,0)
                new_membership+=i*new_mem
        #print((new_membership == 0).sum(dim=0))
        #print((new_membership == 1).sum(dim=0))
        #print((new_membership == 2).sum(dim=0))
        #print((new_membership == 3).sum(dim=0))

    #Final Partitions distribution and entropy calculation
        zeros_dict_,ones_dict_=get(args.grp_parts,new_membership,g.ndata["train_mask"],labels_copy)
        dict_entropy_train,new_total_entropy=entropy(zeros_dict_,ones_dict_)
        print("Final partition entropy:    ",dict_entropy_train)
        print("Total Entropy: "+str(new_total_entropy))
        
        min_total_entropy_list.append(new_total_entropy)
        if min_total_entropy > new_total_entropy:
            min_run=repeat
            min_total_entropy=new_total_entropy
            min_new_membership=copy.deepcopy(new_membership)
    #Gap between max and min entropy
    
        max_value=max(list(dict_entropy_train.values()))
        min_value=min(list(dict_entropy_train.values()))
        print("Entropy Gap between max and min value :       ",max_value-min_value)
        print("Total Time for this run : ", time.time()-loopstart)
        print("...............................................................")
    print("Total Time : ", time.time()-begin_start)
    print(".......................................Final Selection...........................")
    print("Selected Run : ", min_run, "With entropy value" , min_total_entropy)    
    
    print("All values", min_total_entropy_list)
    
    print(".........................new partition call to dump partitions.........................")
    dump_part.improved_partition_graph(g,
            args.grp_parts,
            args.sample_length,
            args.output,
            True,
            new_membership,
            graph_name=args.dataset,
            balance_ntypes=None,
            balance_edges=False,
            num_hops=1,
            return_mapping=False
            )
    #print("Total Time : ", time.time()-runstart)
    print("......................................................................................")

