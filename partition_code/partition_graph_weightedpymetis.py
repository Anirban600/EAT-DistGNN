import argparse
import os
import sys
import time
import newpartitioing_graph
import numpy as np
import torch as th
from torch import linalg as LA

import json
import pymetis
import math

import dgl
import dgl.data

from statistics import mean
from os import path
import sys
from collections import Counter
from ogb.nodeproppred import DglNodePropPredDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from load_graph import load_ogb, load_reddit
from load_graph_custom import load_flicker, load_yelp,load_reddit

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
    args = argparser.parse_args()
    
    # Load Data
    data = DglNodePropPredDataset(name="ogbn-products", root="./products")
    g, labels = data[0]
    #dataset = dgl.data.CoraGraphDataset()
    #g = dataset[0]
   
    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit()
    elif args.dataset == "ogb-product":
        g, _ = load_ogb("ogbn-products")
    elif args.dataset == "ogb-paper100M":
        g, _ = load_ogb("ogbn-papers100M")
    elif args.dataset == "ogb-arxiv":
        g, _ = load_ogb("ogbn-arxiv")
    elif args.dataset == "flickr":
        g, _ = load_flicker()
    elif args.dataset == "yelp":
        g, _ = load_yelp()
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
    
    if args.part_method=="metis":
        print(".................................................Metis partition call....................................")
        start=time.time()
        dgl.distributed.partition_graph(
         g,
         args.dataset,
         args.num_parts,
         args.output,
         part_method=args.part_method,
         balance_ntypes=balance_ntypes,
         balance_edges=args.balance_edges,
         num_trainers_per_machine=args.num_trainers_per_machine,
         )
        print("........................................................")
        print("Total Time : ", time.time()-start)
        
    if args.part_method=="pymetis":
        runstart=time.time()
        print("..............................................................Normalization....................................................................................")
        counter=0
        cnt=0 
        start=time.time()
        # Normalize the feature vectors for all the nodes in the graph
        for u in g.nodes():
            g.ndata['features'][u]=g.ndata['features'][u]/(0.00001+LA.norm(g.ndata['features'][u]))
        print(
             "Normalization of feature vectors {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
        )
        #print(g.ndata['features'][0])
        #print("............................................................................................................................................................")
        print("...............................................................CSC Represenation............................................................................")
        # Get the CSC representation of the graph	    
        iptr, indx, eid = g.adj_sparse('csc')
        #print("Index Pointer ",iptr)
        #print(iptr)
        #print(len(iptr))
        #print("Col Indices ",indx)
        #print(indx)
        #print(len(indx))
        #print("Edge Ids:",eid)
        #print(eid)
        #print(len(eid))
        #adj_list=[]
        #weights=np.empty(len(eid), dtype=float)
        weights=np.zeros(len(eid))
        #weights=[]
        print("..................................................................Loop Start..............................................................................")
        start=time.time()
        for i in range(len(iptr)-1):
            counter+=1
            #print(counter)
            #if counter > 1:
            #    break
            adj_np=indx[iptr[i]:iptr[i+1]]   # Tensor containing neighbors of node i
            edges=eid[iptr[i]:iptr[i+1]]
            #print(edges)
            N=iptr[adj_np+1]-iptr[adj_np]
            cst=(1 - th.exp(-25*th.ones(N.size())/N))
            #print(cst)
            my_f = g.ndata['features'][i]                # Tensor containing feature of node i
            #print(my_f)
            my_nbr_f=g.ndata['features'][adj_np]
            similarity = th.matmul(my_nbr_f,my_f)
            #print(similarity)
            dot_nbr = (similarity+cst)*100   # Tensor of dot products
            weights[iptr[i]:iptr[i+1]]=dot_nbr
            #print(th.from_numpy(np.setdiff1d(edges.numpy(),eid.numpy())))
            #print(set(edges).difference(edg_not_rmv))
            #if set(set(edges).difference(edg_not_rmv)) 
            #if similarity:
            #    new_src.append(i)
            #    new_dst.append(j)
            if counter%10000 == 0:
                print("Counter is", counter)
                print("Time to process", time.time()-start)
        weights = weights.astype(int)
        #print(weights)
        weights[weights<0] = 0
        print("....................................................................Loop end..............................................................................")
        print(
          "Loop {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
        )
        #print(type(weights))
        #print("Length of weights",len(weights))
        if args.part_method=="pymetis":
                #def bias_metis_partition(num_parts, g, sample_length, xadj, adjncy, eweights):
            def bias_metis_partition(num_parts, g, iptr, indx, weights):
                partition_sets = []
                #partition_sets=np.zeros(4)
                start=time.time()
                n_cuts, membership = pymetis.part_graph(num_parts,xadj=iptr, adjncy=indx, eweights=weights)
                print(
                "pymetis takes {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
                )
                print(len(membership))
                for i in range(num_parts):
                    print(i)
                    partition_data = np.argwhere(np.array(membership) == i).ravel()
                    print(len(partition_data))
                    #print(partition_data)
                    partition_sets.append(partition_data.tolist())
                    return [partition_data, partition_sets,membership]

            print("..............................................bias metis call.............................................................................")
            partition_data, partition_sets,membership_data = bias_metis_partition(args.num_parts, g,iptr, indx, weights)
            
            print("...........................................new partition call.................................................................................")
            newpartitioing_graph.improved_partition_graph(g,
            args.num_parts,
            args.sample_length,
            args.output,
            True,
            #partition_sets,
            #partition_data,
            iptr,
            indx,
            weights,
            #node_part_var,
            membership_data,
            #graph_name='test',
            graph_name=args.dataset,
            balance_ntypes=None,
            balance_edges=False,
            num_hops=1,
            return_mapping=False
            )
        print("........................................................")
        print("Total Time : ", time.time()-runstart)
