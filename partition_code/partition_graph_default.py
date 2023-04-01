import argparse
import os
import sys
import time

import numpy as np
import torch as th

import dgl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from load_graph_custom import load_flicker, load_yelp,load_reddit
#from load_graph import load_ogb, load_reddit

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
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

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
    print("...........................start...................................")
    start = time.time()
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
