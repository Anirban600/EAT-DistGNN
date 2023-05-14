import argparse
import os
import sys
import time

import numpy as np
import torch as th
import statistics
import dgl
from collections import Counter
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import copy
import random
from math import log2
import dgl
import dump_part
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dgl.data import FlickrDataset,RedditDataset



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


# load dataset
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
    #print(g.ndata["labels"].numpy())
    #print("CLASSES:      ",np.unique(g.ndata["labels"].numpy()))
    #print("NUMBER:      ",len(np.unique(g.ndata["labels"].numpy())))
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
   

    runstart = time.time()
    classes=len(np.unique(g.ndata["label"].numpy()))
    #print(classes)
    labels_copy = g.ndata["label"]
    #labels_copy[labels_copy==0]=classes
    


# Metis Call
    print("...........................Metis start.............................")
    start = time.time()
    membership=dgl.metis_partition_assignment(g, args.num_parts, balance_ntypes=balance_ntypes, balance_edges=args.balance_edges, mode='k-way', objtype='cut')
    print("Total Metis Time : ", time.time()-start)
    #print(len(membership))
    print("............................Metis end..............................")
    
#To get training label distribution of all partitions
    def get_dist(num_parts,membership,mask,label):
        dict_t={}
        for i in range(num_parts):
             t_=th.where(membership==i,1,0)*mask
             #t_label=th.where(membership==i,1,0)*mask*label
             train_l = label[t_.bool()]
             uni, freq = th.unique(train_l, sorted=True, return_counts=True)
             uni, freq = list(map(int, uni.tolist())), freq.tolist()
             temp={}
             for key, value in zip(uni, freq):
                 temp[key] = value
                 dict_t[i]=temp
        for i in range(num_parts):
            for j in range(0,classes):
                if j not in dict_t[i]:
                    dict_t[i][j] = 0
        #print(dict_train)
        for i in range(num_parts):
            keys = list(dict_t[i].keys())
            values = list(dict_t[i].values())
            sorted_value_index = np.argsort(keys)
            dict_t[i] = {keys[j]: values[j] for j in sorted_value_index}
        #print("....................sorted....................")
        #print(dict_train)
        return dict_t
    

# Entropy of a given label distribution x
    def ent(x,total_nodes):
        entr=0
        prob=np.zeros(classes)
        #total_nodes=sum(x.values())
        for k, i in x.items():
                if i!=0:
                    p=i/total_nodes
                    prob[k-1]=p
                    buffer=-p*np.log2(p)
                    entr=entr+buffer
        return [entr,prob]

# Entropy of all partitions
    def entropy(dict):
        #entropy=0
        dict_entropy={}
        dict_prob_dist={}
        total_node_count={}
        keys=0
        total_part_nodes=0
        for x in dict.values():
            #prob=np.zeros(classes)
            #entropy=0
            #total_nodes=sum(x.values())
            total_part_nodes+=sum(x.values())
            total_node_count[keys]=sum(x.values())
            #dict_entropy[keys],dict_prob_dist[keys]=ent(x,total_node_count[keys])
            dict_entropy[keys],dict_prob_dist[keys]=ent(x,total_node_count[keys])
            #dict_prob_dist[keys]=prob
            keys+=1
        #print("Probability Distribution:",dict_prob_dist)
        #print("Total Nodes::")
        #print(total_part_nodes)
        #print("Entropy Train for each partition: ")
        #print(dict_entropy)
        keys=0
        total_entropy=0
        for x in dict.values():
            #print("No. of nodes in a partition",keys, sum(x.values()))
            total_entropy+=(sum(x.values())/total_part_nodes)*dict_entropy[keys]
            #print(dict_entropy[keys])
            keys+=1
        #print("Total Entropy: "+str(total_entropy))
        keys = list(dict_entropy.keys())
        values = list(dict_entropy.values())
        sorted_value_index = np.argsort(values)
        dict_entropy = {keys[i]: values[i] for i in sorted_value_index}
        #print("Sorted Entropy for each partition: ")
        return [dict_entropy,dict_prob_dist,total_node_count,total_entropy]


# for initial metis partitions
    #print("....................dict train.........................")
    dict_train_labels=get_dist(args.num_parts,membership,g.ndata["train_mask"],labels_copy)
    #print("Training Label Distribution: ",dict_train_labels)
    #print(dict_train_feat)
    #print("....................Entropy Calculation................")
    dict_entropy,train_prob_dist,total_node_count,pre_total_entropy=entropy(dict_train_labels)
    #print("Entropy of each partition: ",dict_entropy)


#get minimum value
    min_total_entropy_list=[]
    min_total_entropy=100
    min_new_membership=th.zeros(len(membership))
    min_run=0
    begin_loop=time.time()
    for repeat in range(args.num_runs):
        print("Run : ",repeat)
        loopstart=time.time()
# initialise the compute nodes 
        keys_val = list(dict_entropy.keys())
        #print(keys_val)
        random.shuffle(keys_val)
        #keys_val=[1, 9, 7, 10, 11, 3, 8, 4, 6, 5, 2, 0]
        #print("Sequence :", keys_val)
        Finalpart={}
        load_partition={}
        Final_part_labels={}
     
    
        for i in range(args.grp_parts):
            load=np.zeros(int(args.num_parts/args.grp_parts))
            load[0]=keys_val[i]
            Final_part_labels[i]=dict_train_labels[keys_val[i]]
            Finalpart[i]=load
            load_partition[i]=int(args.num_parts/args.grp_parts)-1
    
        #print(Finalpart)
        #print(Final_part_labels)
        #print(load_partition)
        
    # cluster remaining partitions to compute nodes
        key_val=keys_val[args.grp_parts:]
        l=list(range(args.grp_parts))
        for k in range(len(key_val)):
            min_inx=0
            min_diff=100
            for x in key_val:
                #print("Check for parttion:   "+str(x))
                for i in l:
                    temp_labels=Counter(Final_part_labels[i])+Counter(dict_train_labels[x])
                    entr_i_and_x,prob3=ent(dict(temp_labels),sum(dict(temp_labels).values()))
                    entr_i,prob1=ent(Final_part_labels[i],sum(Final_part_labels[i].values()))
                    entr_x,prob2=ent(dict_train_labels[x],sum(dict_train_labels[x].values()))
                    diff_ent=entr_i_and_x - entr_i - entr_x
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
            l.remove(min_inx)
            key_val.remove(keep_x)
            if len(l)==0:
                l=list(range(args.grp_parts))
            #print(l)
            #print(Finalpart)
            #print("parttion "+str(keep_x)+" is assigned to group "+str(min_inx))
            #Finalpart[min_inx]=Finalpart[min_inx]+part_mask[keep_x]
            Final_part_labels[min_inx]=dict(Counter(Final_part_labels[min_inx])+Counter(dict_train_labels[keep_x]))
           
            
            #print("Final Assignment: ",load_partition)
           
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
        #print("....................For final training nodes.........................")
        dict_train_final_labels=get_dist(args.grp_parts,new_membership,g.ndata["train_mask"],labels_copy)
        #print("Final partition distribution:    ",dict_train_final_labels)
        #print("....................Entropy Calculation................")
        dict_entropy_train1,train_prob_dist1,train_node_count1,new_total_entropy=entropy(dict_train_final_labels)
        print("Final partition entropy:    ",dict_entropy_train1)
        print("Total Entropy: "+str(new_total_entropy))
        
        min_total_entropy_list.append(new_total_entropy)
        if min_total_entropy > new_total_entropy:
            min_run=repeat
            min_total_entropy=new_total_entropy
            min_new_membership=copy.deepcopy(new_membership)
    #Gap between max and min entropy
        max_value=max(list(dict_entropy_train1.values()))
        min_value=min(list(dict_entropy_train1.values()))
        print("Entropy Gap between max and min value :       ",max_value-min_value)
        print("Standard Deviation: ",statistics.stdev(list(dict_entropy_train1.values())))
        print("Time for this run : ", time.time()-loopstart)
        print("...............................................................")
    print("Total Time : ", time.time()-begin_loop)
    print(".......................................Final Selection...........................")
    print("Selected Run : ", min_run, "With entropy value" , min_total_entropy)    
    print("All values", min_total_entropy_list)
    print(".........................new partition call to dump partitions.........................")
    dump_part.improved_partition_graph(g,
            args.grp_parts,
            args.sample_length,
            args.output,
            True,
            min_new_membership,
            graph_name=args.dataset,
            balance_ntypes=None,
            balance_edges=False,
            num_hops=1,
            return_mapping=False
            )
    #print("Total Time : ", time.time()-runstart)
    print("............................... End .......................................................")
    
