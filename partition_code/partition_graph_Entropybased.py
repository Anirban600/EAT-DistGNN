import argparse
import os
import sys
import time

import numpy as np
import torch as th

import dgl
from collections import Counter

import copy

import random

from math import log2
import dgl
import dump_part

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from load_graph import load_ogb, load_reddit
from load_graph_custom import load_flicker, load_yelp

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
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit()
    elif args.dataset == "ogb-product":
        g, _ = load_ogb("ogbn-products")
    elif args.dataset == "ogb-paper100M":
        g, _ = load_ogb("ogbn-papers100M")
    elif args.dataset == "flicker":
        g, _ = load_flicker()
    elif args.dataset == "yelp":
        g, _ = load_yelp()
    elif args.dataset == "ogb-arxiv":
        g, _ = load_ogb("ogbn-arxiv")
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
    print(g.ndata["labels"].numpy())
    print("CLASSES:      ",np.unique(g.ndata["labels"].numpy()))
    print("NUMBER:      ",len(np.unique(g.ndata["labels"].numpy())))
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
   
    
    cosi = th.nn.CosineSimilarity(dim=1)
    cosir = th.nn.CosineSimilarity(dim=0)
    runstart = time.time()
    classes=len(np.unique(g.ndata["labels"].numpy()))
    print(classes)
    labels_copy = copy.deepcopy(g.ndata["labels"])
    labels_copy[labels_copy==0]=classes
    features=copy.deepcopy(g.ndata["features"])
    print(len(g.ndata["features"][0]))


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
            t_label=th.where(membership==i,1,0)*mask*label
            cnt=Counter()
            for num in t_label.tolist():
                cnt[num]+=1
            temp={}
            for key, value in cnt.items():
                temp[key] = value
            dict_t[i]=temp
            del dict_t[i][0]
        for i in range(num_parts):
            for j in range(1,classes+1,1):
                if j not in dict_t[i]:
                    dict_t[i][j] = 0
        #print(dict_train)
        for i in range(num_parts):
            keys = list(dict_t[i].keys())
            values = list(dict_t[i].values())
            sorted_value_index = np.argsort(keys)
            dict_t[i] = {keys[j]: values[j] for j in sorted_value_index}
        print("....................sorted....................")
        #print(dict_train)
        return dict_t
    #def get_feat():
    def get_feat(num_parts,membership,mask):
        dict_feat={}
        for i in range(num_parts):
            t_feat=th.index_select(features,0,th.nonzero(th.where(membership==i,1,0)*mask, as_tuple=True)[0])
            dict_feat[i]=th.sum(t_feat, 0)
        return dict_feat

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
        print("Total Nodes::")
        print(total_part_nodes)
        #print("Entropy Train for each partition: ")
        #print(dict_entropy)
        keys=0
        total_entropy=0
        for x in dict.values():
            print("No. of nodes in a partition",key, sum(x.values()))
            total_entropy+=(sum(x.values())/total_part_nodes)*dict_entropy[keys]
            #print(dict_entropy[keys])
            keys+=1
        print("Total Entropy: "+str(total_entropy))
        keys = list(dict_entropy.keys())
        values = list(dict_entropy.values())
        sorted_value_index = np.argsort(values)
        dict_entropy = {keys[i]: values[i] for i in sorted_value_index}
        print("Sorted Entropy for each partition: ")
        return [dict_entropy,dict_prob_dist,total_node_count]


# for initial metis partitions
    print("....................dict train.........................")
    dict_train_labels=get_dist(args.num_parts,membership,g.ndata["train_mask"],labels_copy)
    dict_train_feat=get_feat(args.num_parts,membership,g.ndata["train_mask"])
    dict_test_feat=get_feat(args.num_parts,membership,g.ndata["test_mask"])
    print("Training Label Distribution: ",dict_train_labels)
    #print(dict_train_feat)
    print("....................Entropy Calculation................")
    dict_entropy,train_prob_dist,total_node_count=entropy(dict_train_labels)
    print("Entropy of each partition: ",dict_entropy)


# initialise the compute nodes 
    keys_val = list(dict_entropy.keys())
    #print(keys_val)
    random.shuffle(keys_val)
    #keys_val=[4, 11, 7, 2, 5, 3, 8, 1, 10, 6, 9, 0]
    print("Sequence :", keys_val)
    Finalpart={}
    load_partition={}
    Final_part_labels={}
    Final_part_trainfeat={}
    Final_part_testfeat={}
   
    for i in range(args.grp_parts):
        load=np.zeros(int(args.num_parts/args.grp_parts))
        load[0]=keys_val[i]
        Final_part_labels[i]=dict_train_labels[keys_val[i]]
        Final_part_trainfeat[i]=dict_train_feat[keys_val[i]]
        Final_part_testfeat[i]=dict_test_feat[keys_val[i]]
        Finalpart[i]=load
        load_partition[i]=int(args.num_parts/args.grp_parts)-1
   
    print(Finalpart)
    print(Final_part_labels)
    print(load_partition)
    
# cluster remaining partitions to compute nodes
    key_val=keys_val[args.grp_parts:]
    l=list(range(args.grp_parts))
    for k in range(len(key_val)):
        min_inx=0
        min_diff=100
        for x in key_val:
            print("Check for parttion:   "+str(x))
            for i in l:
                temp_labels=Counter(Final_part_labels[i])+Counter(dict_train_labels[x])
                entr2,prob2=ent(dict(temp_labels),sum(dict(temp_labels).values()))
                entr1,prob1=ent(Final_part_labels[i],sum(Final_part_labels[i].values()))
                diff_ent=entr2-entr1
                temp_trainfeat=Final_part_trainfeat[i]+dict_train_feat[x]
                temp_testfeat=Final_part_testfeat[i]+dict_test_feat[x]
                cos_sim=cosir(temp_trainfeat,temp_testfeat)
                diff=diff_ent
                #diff=diff_ent-4*cos_sim
                print("With group "+str(i)+ "  Entropy difference:  "+str(diff_ent)+"Total diff"+str(diff))
                #print("With group "+str(i)+" Feature similarity "+str(cos_sim)+"Entropy difference:  "+str(diff_ent)+"Total diff"+str(diff))
                if min_diff>diff:
                    min_diff=diff
                    min_inx=i
                    keep_x=x
        Finalpart[min_inx][load_partition[min_inx]]=keep_x
        load_partition[min_inx]-=1
        #Finalpart[min_inx][]
        #if load_partition[min_inx]==0:
        #    l.remove(min_inx)
        #if len(l)==0:
        #    l=list(range(args.grp_parts))
        #print(l)
        #print(load_partition)
        #------------------------------------------
        l.remove(min_inx)
        key_val.remove(keep_x)
        if len(l)==0:
            l=list(range(args.grp_parts))
        #print(l)
        print(Finalpart)
        print("parttion "+str(keep_x)+" is assigned to group "+str(min_inx))
        #Finalpart[min_inx]=Finalpart[min_inx]+part_mask[keep_x]
        Final_part_labels[min_inx]=dict(Counter(Final_part_labels[min_inx])+Counter(dict_train_labels[keep_x]))
        Final_part_trainfeat[min_inx]=Final_part_trainfeat[min_inx]+dict_train_feat[keep_x]
        Final_part_testfeat[min_inx]=Final_part_testfeat[min_inx]+dict_test_feat[keep_x]
        
        print("Final Assignment: ",load_partition)
        #print(Final_part_trainfeat)
        
    new_membership=th.zeros(len(membership))
# Membership of compute nodes
    for i in range(args.grp_parts):
        if i !=0:
            new_mem=th.zeros(len(membership))
            for j in list(Finalpart[i]):
                new_mem+=th.where(membership==j,1,0)
            new_membership+=i*new_mem
    print((new_membership == 0).sum(dim=0))
    print((new_membership == 1).sum(dim=0))
    print((new_membership == 2).sum(dim=0))
    print((new_membership == 3).sum(dim=0))

#Final Partitions distribution and entropy calculation
    print("....................dict train.........................")
    dict_train_final_labels=get_dist(args.grp_parts,new_membership,g.ndata["train_mask"],labels_copy)
    print("Final partition distribution:    ",dict_train_final_labels)
    print("....................Entropy Calculation................")
    dict_entropy_train1,train_prob_dist1,train_node_count1=entropy(dict_train_final_labels)
    print("Final partition entropy:    ",dict_entropy_train1)


#Gap between max and min entropy

    max_value=max(list(dict_entropy_train1.values()))
    min_value=min(list(dict_entropy_train1.values()))
    print("Entropy Gap between max and min value :       ",max_value-min_value)


    print("....................dict test.........................")
    dict_test_final_labels=get_dist(args.grp_parts,new_membership,g.ndata["test_mask"],labels_copy)
    #print(dict_test_final_labels)
    print("....................Entropy Calculation................")
    dict_entropy_test,test_prob_dist,test_node_count=entropy(dict_test_final_labels)
    #print(dict_entropy_test)
   
# calculate the kl divergence
    def kl_divergence(p, q,inv_total_node):
        sum=0
        for i in range(len(p)):
            #print("p:   ",i,p[i])
            #
            # print("q:   ",i,q[i])
            if p[i]!=0: 
                if q[i]==0:
                    sum=sum+p[i] * log2(p[i]/inv_total_node)
                else:
                    sum=sum+p[i] * log2(p[i]/q[i])
        return sum
    print(".........................total...........................")
    print(test_node_count)
    print(train_node_count1)
    for i in range(args.grp_parts): 
        p=train_prob_dist1[i]
        q=test_prob_dist[i]
        kl_pq = kl_divergence(p, q,(1/(100*test_node_count[i])))
        print("For Partition :  ",i)
        print('KL(P || Q): %.3f bits' % kl_pq)
        # calculate (Q || P)
        kl_qp = kl_divergence(q, p,(1/(100*train_node_count1[i])))
        print('KL(Q || P): %.3f bits' % kl_qp)
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
    print("Total Time : ", time.time()-runstart)
    print("......................................................................................")
