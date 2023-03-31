import argparse
import socket
import time
import psutil
from contextlib import contextmanager

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
from utils import save_metrics, plot_graphs
# import torch.profiler


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["features"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["labels"][seeds].to(device)
    return batch_inputs, batch_labels


def compute_acc(pred, labels, agg=False):
    """
    Compute the accuracy of prediction given the labels.
    """
    num_machines = dgl.distributed.get_num_client()
    macro_f1 = th.tensor(sklearn.metrics.f1_score(
        labels, pred, average='macro'))
    micro_f1 = th.tensor(sklearn.metrics.f1_score(
        labels, pred, average='micro'))
    ret_mac, ret_mic = macro_f1.item(), micro_f1.item()
    if agg:
        td.all_reduce(macro_f1, op=td.ReduceOp.SUM)
        td.all_reduce(micro_f1, op=td.ReduceOp.SUM)
        avg_macro_f1 = macro_f1 / num_machines
        avg_micro_f1 = micro_f1 / num_machines
        return ret_mac, ret_mic, avg_macro_f1.item(), avg_micro_f1.item()
    return macro_f1.item(), micro_f1.item()


def evaluate(ensemble, g, inputs, labels, nid, batch_size, device, agg, n_classes, report=False, tvt_ids = None):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    if nid == None:
        nid = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
    pred = None
    for model in ensemble:
        model.eval()
        with th.no_grad():
            if pred == None:
                pred = F.softmax(model.inference(
                    g, inputs, nid, batch_size, device), dim=1)
            else:
                pred += F.softmax(model.inference(g, inputs,
                                  nid, batch_size, device), dim=1)
        model.train()
    pred = F.softmax(pred, dim=1)
    pred_l = pred.detach()
    pred = th.argmax(pred, dim=1).detach().numpy()
    if report:
        class_report = []
        for split in tvt_ids:
            class_report.append(classification_report(labels[split].detach().numpy(), pred[split], output_dict=True, zero_division=0))
        temp = [{
            'train': class_report[0],
            'val': class_report[1],
            'test': class_report[2]
        }, pred_l]
        return temp
    return compute_acc(pred[nid], labels[nid].detach().numpy(), agg=agg)


def train_model(model, loss_fcn, g, dataloader, device, metrics, args, val_nid, agg, stopper=None):
    process = psutil.Process()
    iter_tput = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = int(args.num_epochs*args.genper_ratio) if agg else (
        args.num_epochs-int(args.num_epochs*args.genper_ratio))
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
            if step > 0 and step % args.log_every == 0:
                batch_labels = batch_labels.long().detach().numpy()
                batch_pred = th.argmax(batch_pred, dim=1).detach().numpy()
                macro_f1, micro_f1 = compute_acc(batch_pred, batch_labels)
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
                        np.sum(step_time[-args.log_every:])
                    )
                )
                metrics['train_loss'].append(loss.item())
                metrics['train_macf1'].append(macro_f1)
                metrics['train_micf1'].append(micro_f1)
                metrics['train_speed'].append(np.mean(iter_tput[3:]))
                metrics['mem_usage'].append(mem_usage)
            start = time.time()

        toc = time.time()
        # profiler.step()
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
            [model] if (args.standalone or not agg) else [model.module],
            g,
            g.ndata["features"],
            g.ndata["labels"],
            val_nid,
            args.batch_size_eval,
            device,
            agg=agg,
            n_classes=args.n_classes
        )
        val_end = time.time()
        metrics['val_macf1'].append(val_scores[0])
        metrics['val_micf1'].append(val_scores[1])
        print(" Rank {} | Val macro_f1 {:.4f} | Val micro_f1 {:.4f} | val time {:.3f}s".format(
            g.rank(), val_scores[0], val_scores[1], val_end-val_start))
        if(g.rank() == 0 and agg):
            print(" Avg Val macro_f1 {:.4f} | Avg Val micro_f1 {:.4f}".format(
                val_scores[2], val_scores[3]))
        if args.early_stop:
            if stopper.step(val_scores[1], 1-int(agg), model, val_scores[-1]):
                print("Stopping Early")
                break


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    pb = g.get_partition_book()
    No_partition = pb.num_partitions()
    print("Number of partitions: ", No_partition)
    shuffle = True
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    # Define model and optimizer
    general_model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout
    )
    personal_model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout
    )
    cs = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=1,
        correction_adj="DAD",
        num_smoothing_layers=50,
        smoothing_alpha=0.9,
        smoothing_adj="DAD",
        autoscale=True,
        scale=20.0,
    )
    general_model = general_model.to(device)
    personal_model = personal_model.to(device)
    cs = cs.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            general_model = th.nn.parallel.DistributedDataParallel(
                general_model)
        else:
            general_model = th.nn.parallel.DistributedDataParallel(
                general_model, device_ids=[device], output_device=device
            )
    if args.early_stop:
        stopper = EarlyStopping(
            model_save_path=f'{args.metrics_path}/es_checkpoint_{g.rank()}.pt', num_machines=dgl.distributed.get_num_client())
    loss_fcn = FocalLoss(n_classes, gamma=args.gamma)
    loss_fcn = loss_fcn.to(device)

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
    preds = {}

    with general_model.join():
        train_model(general_model, loss_fcn, g, dataloader, device, metrics,
                    args, val_nid, True, stopper if args.early_stop else None)

    if args.early_stop:
        print("Loading the best model after generalization")
        stopper.reset()
        personal_model.load_state_dict(
            th.load(f'{args.metrics_path}/es_checkpoint_{g.rank()}.pt'))
    else:
        personal_model = general_model.module
    
    temp = evaluate([personal_model], g, g.ndata["features"], g.ndata["labels"], None, args.batch_size_eval, device, False, n_classes, True, (train_nid, val_nid, test_nid))
    final_report['GEN'] = temp[0]
    preds['GEN'] = temp[1]

    
    if args.genper_ratio < 1:
        print("Personalization started")
        train_model(personal_model, loss_fcn, g, dataloader, device, metrics,
                    args, val_nid, False, stopper if args.early_stop else None)
        g.barrier()
        if args.early_stop:
            print("Loading the best model after personalization")
            personal_model.load_state_dict(
                th.load(f'{args.metrics_path}/es_checkpoint_{g.rank()}.pt'))
        temp = evaluate([personal_model], g, g.ndata["features"], g.ndata["labels"], None,
                           args.batch_size_eval, device, False, n_classes, True, (train_nid, val_nid, test_nid))
        final_report['PER'] = temp[0]
        preds['PER'] = temp[1]
    if not args.early_stop:
        th.save(personal_model.state_dict(), f'{args.metrics_path}/es_checkpoint_{g.rank()}.pt')
    models = []
    if args.ensemble:
        print("Doing ensemble inference")
        num_machines = dgl.distributed.get_num_client()
        for i in range(num_machines):
            temp_model = DistSAGE(
                in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
            temp_model.load_state_dict(
                th.load(f'{args.metrics_path}/es_checkpoint_'+str(i)+'.pt'))
            models.append(temp_model)
        temp = evaluate([personal_model], g, g.ndata["features"], g.ndata["labels"], None,
                           args.batch_size_eval, device, False, n_classes, True, (train_nid, val_nid, test_nid))
        final_report['EN'] = temp[0]
        preds['EN'] = temp[1]
    else:
        models.append(personal_model)

    if args.c_and_s:
        print("Doing Correct and Smooth")
        train_nid_l = th.from_numpy(np.intersect1d(
            train_nid.numpy(), g.local_partition.ndata['_ID']))
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        with th.no_grad():
            cs_start = time.time()
            y_soft = None
            for model in models:
                model.eval()
                with th.no_grad():
                    if y_soft == None:
                        y_soft = F.softmax(model.inference(
                            g, g.ndata["features"], nodes, args.batch_size_eval, device), dim=1)
                    else:
                        y_soft += F.softmax(model.inference(
                            g, g.ndata["features"], nodes, args.batch_size_eval, device), dim=1)
                model.train()
            y_soft = F.softmax(y_soft, dim=-1)
            y_soft_l = y_soft[g.local_partition.ndata["_ID"]]
            y_soft_l = cs.correct(
                g.local_partition, y_soft_l, g.ndata["labels"][train_nid_l], pb.nid2localnid(train_nid_l, pb.partid))
            y_soft_l = cs.smooth(g.local_partition, y_soft_l, g.ndata["labels"][train_nid_l], pb.nid2localnid(
                train_nid_l, pb.partid))
            y_soft[g.local_partition.ndata["_ID"]] = y_soft_l
            cs_end = time.time()
            train_labels = g.ndata["labels"][train_nid].long().detach().numpy()
            train_pred = th.argmax(y_soft[train_nid], dim=1).detach().numpy()
            val_labels = g.ndata["labels"][val_nid].long().detach().numpy()
            val_pred = th.argmax(y_soft[val_nid], dim=1).detach().numpy()
            test_labels = g.ndata["labels"][test_nid].long().detach().numpy()
            test_pred = th.argmax(y_soft[test_nid], dim=1).detach().numpy()
            (c_val_macro_f1, c_val_micro_f1) = compute_acc(
                val_pred, val_labels, agg=False)
            (c_test_macro_f1, c_test_micro_f1) = compute_acc(
                test_pred, test_labels, agg=False)
            temp = {}
            temp['train'] = classification_report(
                train_labels, train_pred, output_dict=True, zero_division=0)
            temp['val'] = classification_report(
                val_labels, val_pred, output_dict=True, zero_division=0)
            temp['test'] = classification_report(
                test_labels, test_pred, output_dict=True, zero_division=0)
            final_report['CS'] = temp
            preds['CS'] = y_soft
        print(
            "Part {}, Val macro f1 {:.4f}, Val micro f1 {:.4f}".format
            (
                g.rank(), c_val_macro_f1, c_val_micro_f1
            )
        )
        print(
            "Part {}, Test macro f1 {:.4f}, Test micro f1 {:.4f}, c & s time: {:.4f}".format
            (
                g.rank(), c_test_macro_f1, c_test_micro_f1, cs_end-cs_start
            )
        )
    # print(final_report)
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
    print(df.iloc[-1][('support','train','GEN')])
    for l1 in ['train','val','test']:
        for l2 in final_report.keys():
            if l2 != 'GEN':
                df=df.drop(('support',l1,l2), axis = 1)
            else:
                df[('support', l1, 'GEN')] = (df[('support', l1, 'GEN')] /
                                              df.iloc[-1][('support', l1, 'GEN')]) * 100
    df.to_csv(f'{args.metrics_path}/classification_report_{g.rank()}.csv')
    # print(preds)
    for k in preds.keys():
        th.save(preds[k],f'{args.metrics_path}/{k}_pred_{g.rank()}.pt')
    # if(g.rank() == 0):
    #     print("Avg test mac {:.4f}, Avg test mic {:.4f}".format(avg_test_mac, avg_test_mic))
    save_metrics(metrics, args.metrics_path, g.rank())
    plot_graphs(metrics, args.metrics_path, g.rank())


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(
        args.graph_name,
        part_config=args.part_config
    )
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    # del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    td.barrier()
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=41, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="25,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=35)
    # parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
             "of batches to be the same.",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="tensorpipe",
        help="backend net type, 'socket' or 'tensorpipe'",
    )
    parser.add_argument("--genper_ratio", type=float, default=1.0,
                        help="amount of generalization and personalization")
    parser.add_argument("--llambda", type=float,
                        default=0.00001, help="regularization weight")
    parser.add_argument("--metrics_path", type=str, default="/home/vishwesh",
                        help="give paths to store metrics and graphs")
    parser.add_argument('--early_stop', type=int, default=0,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--c_and_s", type=int, default=0,
                        help="whether to apply correct and smooth")
    parser.add_argument("--ensemble", type=int, default=0,
                        help="whether to apply ensemble")
    args = parser.parse_args()

    print(args)
    main(args)
