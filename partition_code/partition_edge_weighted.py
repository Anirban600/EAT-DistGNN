import argparse
import os
import sys
import time
import dump_part
import numpy as np
import torch as th
from torch import linalg as LA
import copy
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import json
import pymetis
import math
import dgl
import dgl.data
from statistics import mean
from os import path
import sys
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dgl.data import FlickrDataset,RedditDataset,YelpDataset
from visualize_partitions import visualize_partitions

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogb-product, ogb-paper100M",
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
    argparser.add_argument('--graph_metadata_path', type=str, default='graph_metadata')
    argparser.add_argument('--c', type=float, default=1.0)
    args = argparser.parse_args()
    
   
    start = time.time()
    
    #Load dataset
    if args.dataset == "reddit":
        dataset = RedditDataset()
        g = dataset[0]
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
        g = dataset[0]
    elif args.dataset == "ogbn-papers100M":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M'))
        g = dataset[0]
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv'))
        g = dataset[0]
    elif args.dataset == "flickr":
        dataset = FlickrDataset()
        g = dataset[0]
    elif args.dataset == "yelp":
        dataset = YelpDataset()
        g = dataset[0]
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
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
       balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
    
    iptr, indx, eid = g.adj_sparse('csc')   
    weights=np.zeros(len(eid))
    
    
    #runstart=time.time()
    #print("..........................Normalization............................")
    counter=0
    cnt=0 
    runstart=time.time()
    # Normalize the feature vectors for all the nodes in the graph
    for u in g.nodes():
        g.ndata['feat'][u]=g.ndata['feat'][u]/(0.00001+LA.norm(g.ndata['feat'][u]))
    #print(
    #         "Normalization of feature vectors {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    #)
       
    #print(".................................CSC Represenation..................................")
    # Get the CSC representation of the graph	    
    
        
    #print("...........................Loop Start.................................")
    #start=time.time()
    for i in range(len(iptr)-1):
            #counter+=1
            adj_np=indx[iptr[i]:iptr[i+1]]   # Tensor containing neighbors of node i
            edges=eid[iptr[i]:iptr[i+1]]
            #print(edges)
            N=iptr[adj_np+1]-iptr[adj_np]
            cst=(1 - th.exp(-25*th.ones(N.size())/N))
            #print(cst)
            my_f = g.ndata['feat'][i]                # Tensor containing feature of node i
            #print(my_f)
            my_nbr_f=g.ndata['feat'][adj_np]
            similarity = th.matmul(my_nbr_f,my_f)
            #print(similarity)
            dot_nbr = (args.c*similarity+cst)*100   # Tensor of dot products
            #print(2*similarity)
            weights[iptr[i]:iptr[i+1]]=dot_nbr
            #if counter%10000 == 0:
                #print("Counter is", counter)
                #print("Time to process", time.time()-start)
    weights = weights.astype(int)
    #print(weights)
    weights[weights<0] = 0
    #print("..........................Loop end...............................")
    #print(
    #      "Loop {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    #    )
    
#def bias_metis_partition(num_parts, g, sample_length, xadj, adjncy, eweights):
    def bias_metis_partition(num_parts, g, iptr, indx, weights):
        #start=time.time()
        n_cuts, membership = pymetis.part_graph(num_parts,xadj=iptr, adjncy=indx, eweights=weights)
        #print(
        #    "pymetis takes {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
        #    )
        print("Total Time : ", time.time()-runstart)
        #print(len(membership))
        return membership

    #print(".........................pymetis call..................................")
            #partition_data, partition_sets,membership_data = bias_metis_partition(args.num_parts, g,iptr, indx, weights)
    membership_data = bias_metis_partition(args.num_parts, g,iptr, indx, weights)
    
    #print(".........................new partition call to dump partitions.........................")
    dump_part.improved_partition_graph(g,
            args.num_parts,
            args.sample_length,
            args.output,
            True,
            membership_data,
            graph_name=args.dataset,
            balance_ntypes=None,
            balance_edges=False,
            num_hops=1,
            return_mapping=False
            )
    #print("Total Time : ", time.time()-runstart)
    #print("......................................................................................")
    visualize_partitions(f"{args.output}/{args.dataset}.json", args.num_parts)