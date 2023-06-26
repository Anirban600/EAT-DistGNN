import argparse
import socket
import time
import psutil
import os
from contextlib import contextmanager
import json

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import torch as th
import torch.distributed as td
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tqdm
import dgl
# import dgl.nn.pytorch as dglnn
import sklearn.metrics
from sklearn.metrics import classification_report
from early_stop import EarlyStopping
from models.correct_and_smooth import CorrectAndSmooth
from models.distsage import DistSAGE
from models.focal_loss import FocalLoss
from models.pick_sampler import LabelBalancedSampler
from utils import save_metrics, plot_graphs, save_tuning_summary
from best_hparams import BEST_HPARAMS
# import torch.profiler

multilabel = {
    'yelp': True,
    'reddit': False,
    'flickr': False,
    'ogbn-products': False,
    'ogbn-papers': False
}

all_nid = None

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["feat"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["label"][seeds].to(device)
    return batch_inputs, batch_labels


def compute_acc(pred, labels, agg=False, report=False, multilabel=False):
    """
    Compute the accuracy of prediction given the labels.
    """
    if multilabel:
        predf = th.where(th.sigmoid(pred) > 0.5, 1, 0).numpy()
    else:
        predf = th.argmax(pred, dim=1).detach().numpy()
    labels = labels.detach().numpy()
    num_machines = dgl.distributed.get_num_client()
    if report:
        return classification_report(labels, predf, output_dict=True, zero_division=0)
    macro_f1 = th.tensor(sklearn.metrics.f1_score(
        labels, predf, average='macro', zero_division=0))
    micro_f1 = th.tensor(sklearn.metrics.f1_score(
        labels, predf, average='micro', zero_division=1))
    ret_mac, ret_mic = macro_f1.item(), micro_f1.item()
    if agg:
        td.all_reduce(macro_f1, op=td.ReduceOp.SUM)
        td.all_reduce(micro_f1, op=td.ReduceOp.SUM)
        avg_macro_f1 = macro_f1 / num_machines
        avg_micro_f1 = micro_f1 / num_machines
        return ret_mac, ret_mic, avg_macro_f1.item(), avg_micro_f1.item()
    return macro_f1.item(), micro_f1.item()


def evaluate(ensemble, g, inputs, labels, nid, batch_size, device, agg, n_classes, multilabel, report=False, tvt_ids = None):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    global all_nid

    ensemble[0].eval()
    with th.no_grad():
        pred = ensemble[0].inference(g, inputs, all_nid, batch_size, device, agg)
    ensemble[0].train()

    if report:
        class_report = []
        for split in tvt_ids:
            class_report.append(compute_acc(pred[split], labels[split], False, True, multilabel=multilabel))
        temp = [{
            'train': class_report[0],
            'val': class_report[1],
            'test': class_report[2]
        }]
        return temp
    return compute_acc(pred[nid], labels[nid], agg=agg, multilabel=multilabel)

def train_model(model, loss_fcn, g, train_nid, device, metrics, args, val_nid, agg, stopper=None, reg_model=None):

    lmbda = th.tensor(args["llambda"], requires_grad=True)
    
    process = psutil.Process()
    A = g.local_partition.adj(scipy_fmt='coo')
    pb = g.get_partition_book()
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    local_train_nid = np.intersect1d(train_nid.numpy(), local_nid)
    halo_train_nid = np.setdiff1d(train_nid.numpy(), local_nid)
    lbs = LabelBalancedSampler(A, g.ndata["label"][local_train_nid].numpy(), pb.nid2localnid(local_train_nid, pb.partid).numpy(), multilabel=multilabel[args['graph_name']])
    probs = lbs.all_probabilities()

    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args["fan_out"].split(",")]
    )
    
    def give_train_data(method):
        # print(local_train_nid.size, probs.size)
        if method == 0:
            sample_idx = train_nid
            args["log_every"] = len(train_nid) // args["batch_size"] - 1
        else:
            ln = min(len(local_train_nid), int(int(len(train_nid)/3)-len(halo_train_nid)//4))
            hn = min(len(halo_train_nid), int(int(len(train_nid)/3)-ln))
            sample_idx = np.random.choice(local_train_nid, size=ln, replace=False, p=probs)
            select_halo_nid = np.random.choice(halo_train_nid, size=hn, replace=False)
            sample_idx = th.cat((th.tensor(select_halo_nid, device=device),th.tensor(sample_idx, device=device)))
            # print(len(sample_idx))
            args["log_every"] = len(sample_idx) // args["batch_size"]
        
        train_dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            sample_idx,
            sampler,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        return train_dataloader

    iter_tput = []
    optimizer = optim.Adam([
        {"params": model.parameters()},
        {"params":[lmbda], "lr": args['lr']}
        ],lr=args['lr'])
    # optimizer = optim.Adam(model.parameters(),lr=args['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    if args['graph_name'] == 'yelp':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
    num_epochs = int(args['num_epochs']*args['genper_ratio']) if agg else (
        args['num_epochs']-int(args['num_epochs']*args['genper_ratio']))
    prob_halo = []
    for epoch in range(num_epochs):
        tic = time.time()

        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []
        dataloader = give_train_data(args["sampler"])
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

            tic_step = time.time()
            sample_time += tic_step - start
            # fetch features/labels
            batch_inputs, batch_labels = load_subtensor(
                g, seeds, input_nodes, "cpu"
            )
            batch_labels = batch_labels.long()
            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            # move to target device
            blocks = [block.to(device) for block in blocks]
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()
            optimizer.zero_grad()
            if reg_model != None:
                #personal weight regularization
                regularization = 0
                # Iterate over the parameters of both models
                for p1, p2 in zip(model.parameters(), reg_model.parameters()):
                    # Add the squared difference of each parameter to the diff
                    regularization += th.sum((p2 - p1) ** 2)
                # if g.rank() == 2:
                #     print(loss, regularization, args['llambda'])
                lmbda = th.clamp(lmbda, 0.0, 1.0)
                loss = loss + lmbda*regularization
                # if g.rank() == 2:
                #     print(loss)
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            optimizer.step()
            update_time += time.time() - compute_end

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(
                len(blocks[-1].dstdata[dgl.NID]) / (step_t+0.000001))
            if (step > 0 and step % args['log_every'] == 0):
                macro_f1, micro_f1 = compute_acc(batch_pred, batch_labels, multilabel=multilabel[args['graph_name']])
                mem_usage = process.memory_info().rss / (1024 ** 2)
                print(
                    "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train macro_f1 {:.4f} | Train micro_f1 {:.4f} | Speed (samples/sec) {:.4f} | CPU {:.4f} MB | time {:.3f} s".format(
                        g.rank(),
                        epoch,
                        step,
                        loss.item(),
                        macro_f1,
                        micro_f1,
                        np.mean(iter_tput[3:]),
                        mem_usage,
                        np.sum(step_time[-args['log_every']:])
                    )
                )
                metrics['train_loss'].append(loss.item())
                metrics['train_macf1'].append(macro_f1)
                metrics['train_micf1'].append(micro_f1)
                metrics['train_speed'].append(np.mean(iter_tput[3:]))
                metrics['mem_usage'].append(mem_usage)
            start = time.time()

        toc = time.time()
        scheduler.step()
        metrics['train_time'].append(toc-tic)
        print(
            "Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, "
            "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
            "#inputs: {}".format(
                g.rank(),
                toc - tic,
                sample_time,
                forward_time,
                backward_time,
                update_time,
                num_seeds,
                num_inputs,
            )
        )
        val_start = time.time()
        val_scores = evaluate(
            [model] if (args['standalone'] or not agg) else [model.module],
            g,
            g.ndata["feat"],
            g.ndata["label"],
            val_nid,
            args['batch_size_eval'],
            device,
            agg=agg,
            n_classes=args['n_classes'],
            multilabel=multilabel[args['graph_name']]
        )
        val_end = time.time()
        metrics['val_macf1'].append(val_scores[0])
        metrics['val_micf1'].append(val_scores[1])
        print(" Rank {} | Val macro_f1 {:.4f} | Val micro_f1 {:.4f} | val time {:.3f}s".format(
            g.rank(), val_scores[0], val_scores[1], val_end-val_start))
        if(g.rank() == 0 and agg):
            print(" Avg Val macro_f1 {:.4f} | Avg Val micro_f1 {:.4f}".format(
                val_scores[2], val_scores[3]))
        if args['early_stop']:
            if stopper.step(val_scores[args["stopping_criteria"]], 1-int(agg), model, val_scores[-2+args["stopping_criteria"]]):
                print("Stopping Early")
                break

        print("rank:", g.rank(), "Final lambda:", lmbda)
    
def calc_entropy(labels):
    # Count the frequency of each label
    counts = th.bincount(labels)
    # Divide by the length of the tensor to get probabilities
    probs = counts.float() / labels.shape[0]
    return th.sum(th.special.entr(probs))

def split_nodes(g, force_even):
    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=force_even,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=force_even,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=force_even,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=force_even
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=force_even
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=force_even
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            pb.partid,
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )

    return train_nid, val_nid, test_nid

def run(args, device, data):
    # Unpack data
    global all_nid
    in_feats, n_classes, g = data
    pb = g.get_partition_book()
    No_partition = pb.num_partitions()
    print("Number of partitions: ", No_partition)
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    # if not args["standalone"]:
    #     if args["num_gpus"] == -1:
    #         general_model = th.nn.parallel.DistributedDataParallel(
    #             general_model)
    #     else:
    #         general_model = th.nn.parallel.DistributedDataParallel(
    #             general_model, device_ids=[device], output_device=device
    #         )
    if args["early_stop"]:
        stopper = EarlyStopping(
            model_save_path=f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt', num_machines=dgl.distributed.get_num_client(), patience = args["early_stop"]) 

    metrics = {
        'train_loss': [],
        'train_macf1': [],
        'train_micf1': [],
        'train_speed': [],
        'mem_usage': [],
        'train_time': [],
        'val_macf1': [],
        'val_micf1': []
    }
    final_report = {}

    # sampler = dgl.dataloading.NeighborSampler(
    #     [int(fanout) for fanout in args["fan_out"].split(",")]
    # )
    # dataloader = dgl.dataloading.DistNodeDataLoader(
    #     g,
    #     train_nid,
    #     sampler,
    #     batch_size=args["batch_size"],
    #     shuffle=shuffle,
    #     drop_last=False,
    # )

    ####################################Generalization###################################
    gen_start = time.time()
    if args["genper_ratio"] > 0:
        print("Generalization started")
        if args['graph_name'] in BEST_HPARAMS:
            print("Loading Best Hyperparamters")
            temp = BEST_HPARAMS[args['graph_name']]['g']
            for hp in temp.keys():
                args[hp] = temp[hp]

        loss_fcn = FocalLoss(n_classes, gamma=args["gamma"], multilabel=multilabel[args['graph_name']])
        # loss_fcn = th.nn.CrossEntropyLoss()
        loss_fcn = loss_fcn.to(device)
        train_nid, val_nid, test_nid = split_nodes(g, True)
        all_nid = th.cat((train_nid, val_nid, test_nid))
        args['entropy'] = calc_entropy(g.ndata['label'][train_nid])
        print("Rank", g.rank(), "Entropy", args['entropy'])
        
        general_model = DistSAGE(
            in_feats,
            args["num_hidden"],
            n_classes,
            args["num_layers"],
            F.relu,
            args["dropout"]
        )
        general_model = general_model.to(device)
        general_model = th.nn.parallel.DistributedDataParallel(general_model)


        with general_model.join():

            train_model(general_model, loss_fcn, g, train_nid, device, metrics,
                        args, val_nid, True, stopper if args["early_stop"] else None)
        
        if args["early_stop"]:
            print("Loading the best model after generalization")
            stopper.reset()
            general_model.module.load_state_dict(
                th.load(f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt'))
        else:
            th.save(general_model.module.state_dict(), f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt')
        
        gen_eval_start = time.time()
        temp = evaluate([general_model.module], g, g.ndata["feat"], g.ndata["label"], None, args["batch_size_eval"], device, True, n_classes, multilabel[args['graph_name']], True, (train_nid, val_nid, test_nid))
        gen_eval_end = time.time()
        final_report['GEN'] = temp[0]
    gen_end = time.time()
    ####################################Personalization###################################
    per_start = time.time()
    if args["genper_ratio"] < 1:
        print("Personalization started")
        if args['graph_name'] in BEST_HPARAMS:
            print("Loading Best Hyperparamters")
            temp = BEST_HPARAMS[args['graph_name']][g.rank()]
            for hp in temp.keys():
                args[hp] = temp[hp]
        if args['graph_name'] == 'yelp':
            args['lr'] *= 0.5
        loss_fcn = FocalLoss(n_classes, gamma=args["gamma"], multilabel=multilabel[args['graph_name']])
        # loss_fcn = th.nn.CrossEntropyLoss()
        loss_fcn = loss_fcn.to(device)
        # args["log_every"] = len(train_nid) // args["batch_size"] - 1
        train_nid, val_nid, test_nid = split_nodes(g, False)
        all_nid = th.cat((train_nid, val_nid, test_nid))
        args['entropy'] = calc_entropy(g.ndata['label'][train_nid])
        print("Rank", g.rank(), "Entropy", args['entropy'])

        personal_model = DistSAGE(
            in_feats,
            args["num_hidden"],
            n_classes,
            args["num_layers"],
            F.relu,
            args["dropout"]
        )
        personal_model = personal_model.to(device)

        if args["genper_ratio"] != 0:
            print("Initializing personal model with general model")
            personal_model.load_state_dict(
                    th.load(f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt'))
            
        train_model(personal_model, loss_fcn, g, train_nid, device, metrics,
                    args, val_nid, False, stopper if args["early_stop"] else None, general_model.module if args["genper_ratio"] != 0 else None)
        if args["early_stop"]:
            print("Loading the best model after personalization")
            personal_model.load_state_dict(
                th.load(f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt'))
        else:
            th.save(personal_model.state_dict(), f'{args["metrics_path"]}/es_checkpoint_{g.rank()}.pt')
    
        g.barrier()
        per_eval_start = time.time()
        temp = evaluate([personal_model], g, g.ndata["feat"], g.ndata["label"], None, args["batch_size_eval"], device, True, n_classes, multilabel[args['graph_name']], True, (train_nid, val_nid, test_nid))
        per_eval_end = time.time()
        g.barrier()
        final_report['PER'] = temp[0]
    per_end = time.time()
    acc_metric = 'micro avg' if multilabel[args['graph_name']] else 'accuracy'
    gen_train_time = 0 if args['genper_ratio']==0 else gen_end-gen_start - (gen_eval_end-gen_eval_start)
    per_train_time = 0 if args['genper_ratio']==1 else per_end-per_start - (per_eval_end-per_eval_start)
    gen_eval_time = 0 if args['genper_ratio']==0 else gen_eval_end-gen_eval_start
    per_eval_time = 0 if args['genper_ratio']==1 else per_eval_end-per_eval_start
    summmary_metrics = {
        'best_acc': 0,
        'best_mac': 0,
        'best_wgt':0,
        'gen_train_time': gen_train_time,
        'per_train_time': per_train_time,
        'train_time': gen_train_time+per_train_time,
        'gen_eval_time': gen_eval_time,
        'per_eval_time': per_eval_time,
        'eval_time': gen_eval_time+per_eval_time,
        'wall_clock_train_time': gen_train_time+per_train_time,
        'wall_clock_time': gen_end-gen_start+per_end-per_start
    }

    if 'GEN' in final_report:
        if multilabel[args['graph_name']]:
            summmary_metrics['best_acc'] = final_report['GEN']['test'][acc_metric]['f1-score']
        else:
            summmary_metrics['best_acc'] = final_report['GEN']['test'][acc_metric]
        summmary_metrics['best_mac'] = final_report['GEN']['test']['macro avg']['f1-score']
        summmary_metrics['best_wgt'] = final_report['GEN']['test']['weighted avg']['f1-score']
    
    if 'PER' in final_report:
        acc_metric = 'micro avg' if multilabel[args['graph_name']] else 'accuracy'
        if multilabel[args['graph_name']]:
            if summmary_metrics['best_acc'] < final_report['PER']['test'][acc_metric]['f1-score']:
                summmary_metrics['best_acc'] = final_report['PER']['test'][acc_metric]['f1-score']
        else:
            if summmary_metrics['best_acc'] < final_report['PER']['test'][acc_metric]:
                summmary_metrics['best_acc'] = final_report['PER']['test'][acc_metric]

        if summmary_metrics['best_mac'] < final_report['PER']['test']['macro avg']['f1-score']:
            summmary_metrics['best_mac'] = final_report['PER']['test']['macro avg']['f1-score']

        if summmary_metrics['best_wgt'] < final_report['PER']['test']['weighted avg']['f1-score']:
            summmary_metrics['best_wgt'] = final_report['PER']['test']['weighted avg']['f1-score']

    # print(summmary_metrics)

    for k in summmary_metrics.keys():
        if 'time' in k and 'wall' not in k:
            temp = th.tensor(summmary_metrics[k])
            td.all_reduce(temp, op=td.ReduceOp.MAX)
            summmary_metrics[k] = temp.item()
        else:
            temp = th.tensor(summmary_metrics[k])
            td.all_reduce(temp, op=td.ReduceOp.SUM)
            if 'best' in k:
                temp/=4
            summmary_metrics[k] = temp.item()
        # summmary_metrics[k] = summmary_metrics[k].item()

    if(g.rank()==0):
        with open(f'{args["metrics_path"]}/summary.json', "w") as outfile:
            json.dump(summmary_metrics, outfile)

    # print(metrics)
    iterables = [['support', 'precision', 'f1-score', 'recall'], ['train', 'val', 'test'], [k for k in final_report.keys()]]
    index = pd.MultiIndex.from_product(iterables)
    final_dict = {}
    for l1 in final_report.keys():
        for l2 in final_report[l1].keys():
            for l3 in final_report[l1][l2].keys():
                if l3 not in final_dict:
                    final_dict[l3] = {}
                if l3 == 'accuracy':
                    # print(l1,l2,l3, final_report[l1][l2][l3])
                    final_dict[l3][('precision', l2, l1)] = final_report[l1][l2][l3]
                    final_dict[l3][('recall', l2, l1)] = final_report[l1][l2][l3]
                    final_dict[l3][('f1-score', l2, l1)] = final_report[l1][l2][l3]
                    continue
                for l4 in final_report[l1][l2][l3].keys():
                    if l4 == 'support' and l1 != 'GEN':
                        continue
                    final_dict[l3][(l4, l2, l1)] = final_report[l1][l2][l3][l4]


    df = pd.DataFrame(final_dict, index=index).transpose()
    # print(df.iloc[-1][('support','train','GEN')])
    for l1 in ['train','val','test']:
        for l2 in final_report.keys():
            temp = list(final_report.keys())[0]
            if l2 != temp:
                df=df.drop(('support',l1,l2), axis = 1)
            else:
                df[('support', l1, temp)] = (df[('support', l1, temp)] /
                                              df.iloc[-2][('support', l1, temp)]) * 100
    df.to_csv(f'{args["metrics_path"]}/classification_report_{g.rank()}.csv')
    with open(f'{args["metrics_path"]}/classification_report_{g.rank()}.json', "w") as outfile:
        json.dump(final_report, outfile)
    # if(g.rank() == 0):
    #     print("Avg test mac {:.4f}, Avg test mic {:.4f}".format(avg_test_mac, avg_test_mic))
    save_metrics(metrics, args["metrics_path"], g.rank())
    plot_graphs(metrics, args["metrics_path"], g.rank())


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    args = vars(args)
    print(args)
    dgl.distributed.initialize(args["ip_config"], net_type=args["net_type"])
    if not args["standalone"]:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args["backend"])
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(
        args["graph_name"],
        part_config=args["part_config"]
    )
    print(socket.gethostname(), "rank:", g.rank())
    
    # del local_nid
    if args["num_gpus"] == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args["num_gpus"]
        device = th.device("cuda:" + str(dev_id))
    n_classes = args["n_classes"]
    if n_classes == 0:
        labels = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["feat"].shape[1]
    data = in_feats, n_classes, g
    if args["tune"]:
        tune_hyperparams(args, device, data)
    else:
        run(args, device, data)
    td.barrier()
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration" )
    parser.add_argument("--part_config", type=str, help="The path to the partition config file" )
    parser.add_argument("--n_classes", type=int, default=40, help="the number of classes" )
    parser.add_argument("--backend", type=str, default="gloo", help="pytorch distributed backend", )
    parser.add_argument("--num_gpus", type=int, default=-1, help="the number of GPU device. Use -1 for CPU training", )
    parser.add_argument("--tune", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="25,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=10)
    # parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--local_rank", type=int, help="get rank of the process" )
    parser.add_argument("--standalone", action="store_true", help="run in the standalone mode" )
    parser.add_argument("--pad-data", default=False, action="store_true", help="Pad train nid to the same length across machine, to ensure num of batches to be the same.", )
    parser.add_argument( "--net_type", type=str, default="tensorpipe", help="backend net type, 'socket' or 'tensorpipe'", )
    parser.add_argument("--genper_ratio", type=float, default=1.0, help="amount of generalization and personalization")
    parser.add_argument("--llambda", type=float, default=0.00001, help="regularization weight")
    parser.add_argument("--metrics_path", type=str, default="/home/vishwesh", help="give paths to store metrics and graphs")
    parser.add_argument('--early_stop', type=int, default=0, help="indicates whether to use early stop and patience or not")
    parser.add_argument('--stopping_criteria', type=int, default=1, help="criteria on which early stop is applied")
    parser.add_argument("--sampler", type=int, default=0, help="which sampler to use")
    args = parser.parse_args()

    print(args)
    main(args)

