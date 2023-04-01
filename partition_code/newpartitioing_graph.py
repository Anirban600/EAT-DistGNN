import pymetis
import numpy as np
import json
import os
import scipy.sparse as spp
import dgl
import torch as th
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict

import torch

from dgl.utils.internal import toindex
#import partition_graph
from os import path
from scipy.sparse import csr_matrix, lil_matrix


import torch.nn.functional as F
from dgl import backend as P
from dgl.base import NID, EID, NTYPE, ETYPE, dgl_warning
from dgl.convert import to_homogeneous
from dgl.partition import partition_graph_with_halo, metis_partition_assignment
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors

#from .. import backend as F
#from ..base import NID, EID, NTYPE, ETYPE, dgl_warning
#from ..convert import to_homogeneous
#from ..random import choice as random_choice
#from ..data.utils import load_graphs, save_graphs, load_tensors, save_tensors
#from ..partition import metis_partition_assignment, partition_graph_with_halo, get_peak_mem
#from .graph_partition_book import BasicPartitionBook, RangePartitionBook

def _get_inner_node_mask(graph, ntype_id):
    if NTYPE in graph.ndata:
        dtype = P.dtype(graph.ndata['inner_node'])
        return graph.ndata['inner_node'] * P.astype(graph.ndata[NTYPE] == ntype_id, dtype) == 1
    else:
        return graph.ndata['inner_node'] == 1

def _get_inner_edge_mask(graph, etype_id):
    if ETYPE in graph.edata:
        dtype = P.dtype(graph.edata['inner_edge'])
        return graph.edata['inner_edge'] * P.astype(graph.edata[ETYPE] == etype_id, dtype) == 1
    else:
        return graph.edata['inner_edge'] == 1

def _get_orig_ids(g, sim_g, reshuffle, orig_nids, orig_eids):
    '''Convert/construct the original node IDs and edge IDs.

    It handles multiple cases:
     * If the graph has been reshuffled and it's a homogeneous graph, we just return
       the original node IDs and edge IDs in the inputs.
     * If the graph has been reshuffled and it's a heterogeneous graph, we need to
       split the original node IDs and edge IDs in the inputs based on the node types
       and edge types.
     * If the graph is not shuffled, the original node IDs and edge IDs don't change.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    reshuffle : bool
        Whether the input graph is reshuffled during partitioning.
    orig_nids : tensor or None
        The original node IDs after the input graph is reshuffled.
    orig_eids : tensor or None
        The original edge IDs after the input graph is reshuffled.

    Returns
    -------
    tensor or dict of tensors, tensor or dict of tensors
    '''
    is_hetero = len(g.etypes) > 1 or len(g.ntypes) > 1
    if reshuffle and is_hetero:
        # Get the type IDs
        orig_ntype = P.gather_row(sim_g.ndata[NTYPE], orig_nids)
        orig_etype = P.gather_row(sim_g.edata[ETYPE], orig_eids)
        # Mapping between shuffled global IDs to original per-type IDs
        orig_nids = P.gather_row(sim_g.ndata[NID], orig_nids)
        orig_eids = P.gather_row(sim_g.edata[EID], orig_eids)
        orig_nids = {ntype: P.boolean_mask(orig_nids, orig_ntype == g.get_ntype_id(ntype)) \
                for ntype in g.ntypes}
        orig_eids = {etype: P.boolean_mask(orig_eids, orig_etype == g.get_etype_id(etype)) \
                for etype in g.etypes}
    elif not reshuffle and not is_hetero:
        orig_nids = P.arange(0, sim_g.number_of_nodes())
        orig_eids = P.arange(0, sim_g.number_of_edges())
    elif not reshuffle:
        orig_nids = {ntype: P.arange(0, g.number_of_nodes(ntype)) for ntype in g.ntypes}
        orig_eids = {etype: P.arange(0, g.number_of_edges(etype)) for etype in g.etypes}
    return orig_nids, orig_eids


def improved_partition_graph(g, num_parts, sample_length, out_path, reshuffle, xadj, adjncy, eweights, node_part_var, graph_name = 'test',balance_ntypes=None, balance_edges=False, num_hops=1, return_mapping=True):
    #partitions = []
    #partition_sets = []


    def get_homogeneous(g, balance_ntypes):
        if len(g.etypes) == 1:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        elif isinstance(balance_ntypes, dict):
            # Here we assign node types for load balancing.
            # The new node types includes the ones provided by users.
            nuNrtition_setsm_ntypes = 0
            for key in g.ntypes:
                if key in balance_ntypes:
                    g.nodes[key].data['bal_ntype'] = P.astype(balance_ntypes[key],
                                                              P.int32) + num_ntypes
                    uniq_ntypes = F.unique(balance_ntypes[key])
                    assert np.all(F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes)))
                    num_ntypes += len(uniq_ntypes)
                else:
                    g.nodes[key].data['bal_ntype'] = P.ones((g.number_of_nodes(key),), F.int32,
                                                            F.cpu()) * num_ntypes
                    num_ntypes += 1
            sim_g = to_homogeneous(g, ndata=['bal_ntype'])
            bal_ntypes = sim_g.ndata['bal_ntype']
            print('The graph has {} node types and balance among {} types'.format(
                len(g.ntypes), len(F.unique(bal_ntypes))))
        else:
            sim_g = to_homogeneous(g)
            bal_ntypes = sim_g.ndata[NTYPE]
        return sim_g, bal_ntypes
    
    if num_parts == 1:
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)

        node_parts = P.zeros((sim_g.number_of_nodes(),), P.int64, P.cpu())
        parts = {0: sim_g.clone()}
        orig_nids = parts[0].ndata[NID] = P.arange(0, sim_g.number_of_nodes())
        orig_eids = parts[0].edata[EID] = P.arange(0, sim_g.number_of_edges())
        # For one partition, we don't really shuffle nodes and edges. We just need to simulate
        # it and set node data and edge data of orig_id.
        if reshuffle:
            parts[0].ndata['orig_id'] = orig_nids
            parts[0].edata['orig_id'] = orig_eids
        if return_mapping:
            orig_nids, orig_eids = _get_orig_ids(g, sim_g, False, orig_nids, orig_eids)
        parts[0].ndata['inner_node'] = P.ones((sim_g.number_of_nodes(),), P.int8, P.cpu())
        parts[0].edata['inner_edge'] = P.ones((sim_g.number_of_edges(),), P.int8, P.cpu())
    else:
        sim_g, _ = get_homogeneous(g, balance_ntypes)
        # node_parts = np.random.choice(num_parts, sim_g.number_of_nodes())
        node_parts = th.tensor(node_part_var, dtype=torch.int64)
    start = time.time()
    parts, orig_nids, orig_eids = partition_graph_with_halo(sim_g, node_parts, num_hops, reshuffle=reshuffle)
    if return_mapping:
            orig_nids, orig_eids = _get_orig_ids(g, sim_g, reshuffle, orig_nids, orig_eids)
    
    # Let's calculate edge assignment.
    if not reshuffle:
        start = time.time()
        # We only optimize for reshuffled case. So it's fine to use int64 here.
        edge_parts = np.zeros((g.number_of_edges(),), dtype=np.int64) - 1
        for part_id in parts:
            part = parts[part_id]
            # To get the edges in the input graph, we should use original node IDs.
            local_edges = P.boolean_mask(part.edata[EID], part.edata['inner_edge'])
            edge_parts[P.asnumpy(local_edges)] = part_id
        print('Calculate edge assignment: {:.3f} seconds'.format(time.time() - start))

    if not reshuffle:
        node_part_file = os.path.join(out_path, "node_map")
        edge_part_file = os.path.join(out_path, "edge_map")
        np.save(node_part_file, P.asnumpy(node_parts), allow_pickle=False)
        np.save(edge_part_file, edge_parts, allow_pickle=False)
        node_map_val = node_part_file + ".npy"
        edge_map_val = edge_part_file + ".npy"
    else:
        if num_parts > 1:
            node_map_val = {}
            edge_map_val = {}
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                val = []
                node_map_val[ntype] = []
                for i in parts:
                    inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                    val.append(P.as_scalar(P.sum(P.astype(inner_node_mask, P.int64), 0)))
                    inner_nids = P.boolean_mask(parts[i].ndata[NID], inner_node_mask)
                    node_map_val[ntype].append([int(P.as_scalar(inner_nids[0])),
                                                int(P.as_scalar(inner_nids[-1])) + 1])
                val = np.cumsum(val).tolist()
                assert val[-1] == g.number_of_nodes(ntype)
            
            for etype in g.etypes:
                etype_id = g.get_etype_id(etype)
                val = []
                edge_map_val[etype] = []
                for i in parts:
                    inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                    val.append(P.as_scalar(P.sum(P.astype(inner_edge_mask, P.int64), 0)))
                    inner_eids = np.sort(P.asnumpy(P.boolean_mask(parts[i].edata[EID],
                                                                  inner_edge_mask)))
                    edge_map_val[etype].append([int(inner_eids[0]), int(inner_eids[-1]) + 1])
                val = np.cumsum(val).tolist()
                assert val[-1] == g.number_of_edges(etype)
        else:
            node_map_val = {}
            edge_map_val = {}
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                inner_node_mask = _get_inner_node_mask(parts[0], ntype_id)
                inner_nids = P.boolean_mask(parts[0].ndata[NID], inner_node_mask)
                node_map_val[ntype] = [[int(P.as_scalar(inner_nids[0])),
                                        int(P.as_scalar(inner_nids[-1])) + 1]]
            for etype in g.etypes:
                etype_id = g.get_etype_id(etype)
                inner_edge_mask = _get_inner_edge_mask(parts[0], etype_id)
                inner_eids = P.boolean_mask(parts[0].edata[EID], inner_edge_mask)
                edge_map_val[etype] = [[int(P.as_scalar(inner_eids[0])),
                                        int(P.as_scalar(inner_eids[-1])) + 1]]

        # Double check that the node IDs in the global ID space are sorted.
        for ntype in node_map_val:
            val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
            assert np.all(val[:-1] <= val[1:])
        for etype in edge_map_val:
            val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
            assert np.all(val[:-1] <= val[1:])
    
    ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype:g.get_etype_id(etype) for etype in g.etypes}
    part_metadata = {'graph_name': graph_name,
                  'num_nodes': g.number_of_nodes(),
                  'num_edges': g.number_of_edges(),
                  # 'part_method': part_method,
                  'num_parts': num_parts,
                  # 'halo_hops': num_hops,
                  'node_map': node_map_val,
                  'edge_map': edge_map_val,
                  'ntypes': ntypes,
                  'etypes': etypes
                  }

    for i in range(num_parts):
        part = parts[i]
        #partitions.append(partition_data)
        #partition_sets.append(set(partition_data))
        #print()
        os.makedirs(out_path, mode=0o775, exist_ok=True)
        out_path = os.path.abspath(out_path)
        start = time.time()
        #ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
        #etypes = {etype:g.get_etype_id(etype) for etype in g.etypes}
        #part_metadata = {'graph_name': graph_name,
        #         'num_nodes': g.number_of_nodes(),
        #         'num_edges': g.number_of_edges(),
        #         # 'part_method': part_method,
        #         'num_parts': num_parts,
        #         # 'halo_hops': num_hops,
        #         'node_map': node_map_val,
        #         'edge_map': edge_map_val,
        #         'ntypes': ntypes,
        #         'etypes': etypes
        #         }

        # # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                # To get the edges in the input graph, we should use original node IDs.
                # Both orig_id and NID stores the per-node-type IDs.
                ndata_name = 'orig_id' if reshuffle else NID
                inner_node_mask = _get_inner_node_mask(part, ntype_id)
                # This is global node IDs.
                local_nodes = P.boolean_mask(part.ndata[ndata_name], inner_node_mask)
                # local_nodes = partition_sets
                if len(g.ntypes) > 1:
                    # If the input is a heterogeneous graph.
                    local_nodes = P.gather_row(sim_g.ndata[NID], local_nodes)
                    print('part {} has {} nodes of type {} and {} are inside the partition'.format(
                        i, P.as_scalar(F.sum(part.ndata[NTYPE] == ntype_id, 0)),
                        ntype, len(local_nodes)))
                else:
                    print('part {} has {} nodes and {} are inside the partition'.format(
                        i, part.number_of_nodes(), len(local_nodes)))
                    # print('part {} has {} nodes and {} are inside the partition'.format(
                    #     i, len(number_node), len(list(partition_data))))

                for name in g.nodes[ntype].data:
                    if name in [NID, 'inner_node']:
                        continue
                    node_feats[ntype + '/' + name] = P.gather_row(g.nodes[ntype].data[name],
                                                                  local_nodes)

            for etype in g.etypes:
                etype_id = g.get_etype_id(etype)
                edata_name = 'orig_id' if reshuffle else EID
                inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                # This is global edge IDs.
                local_edges = P.boolean_mask(part.edata[edata_name], inner_edge_mask)
                if len(g.etypes) > 1:
                    local_edges = P.gather_row(sim_g.edata[EID], local_edges)
                    print('part {} has {} edges of type {} and {} are inside the partition'.format(
                        part_id, P.as_scalar(F.sum(part.edata[ETYPE] == etype_id, 0)),
                        etype, len(local_edges)))
                else:
                    print('part {} has {} edges and {} are inside the partition'.format(
                        i, len(local_edges), len(local_edges)))
                # tot_num_inner_edges += len(local_edges)

                for name in g.edges[etype].data:
                    if name in [EID, 'inner_edge']:
                        continue
                    edge_feats[etype + '/' + name] = P.gather_row(g.edges[etype].data[name],
                                                                  local_edges)
        else:
            for ntype in g.ntypes:
                if reshuffle and len(g.ntypes) > 1:
                    ndata_name = 'orig_id'
                    ntype_id = g.get_ntype_id(ntype)
                    inner_node_mask = _get_inner_node_mask(part, ntype_id)
                    # This is global node IDs.
                    local_nodes = P.boolean_mask(part.ndata[ndata_name], inner_node_mask)
                    local_nodes = P.gather_row(sim_g.ndata[NID], local_nodes)
                elif reshuffle:
                    local_nodes = sim_g.ndata[NID]
                for name in g.nodes[ntype].data:
                    if name in [NID, 'inner_node']:
                        continue
                    if reshuffle:
                        node_feats[ntype + '/' + name] = P.gather_row(g.nodes[ntype].data[name],
                                                                      local_nodes)
                    else:
                        node_feats[ntype + '/' + name] = g.nodes[ntype].data[name]
            for etype in g.etypes:
                if reshuffle and len(g.etypes) > 1:
                    edata_name = 'orig_id'
                    etype_id = g.get_etype_id(etype)
                    inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                    # This is global edge IDs.
                    local_edges = P.boolean_mask(part.edata[edata_name], inner_edge_mask)
                    local_edges = P.gather_row(sim_g.edata[EID], local_edges)
                elif reshuffle:
                    local_edges = sim_g.edata[EID]
                for name in g.edges[etype].data:
                    if name in [EID, 'inner_edge']:
                        continue
                    if reshuffle:
                        edge_feats[etype + '/' + name] = P.gather_row(g.edges[etype].data[name],
                                                                      local_edges)
                    else:
                        edge_feats[etype + '/' + name] = g.edges[etype].data[name]

        part_dir = os.path.join(out_path, "part" + str(i))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(i)] = {
            'node_feats': os.path.relpath(node_feat_file, out_path),
            'edge_feats': os.path.relpath(edge_feat_file, out_path),
            'part_graph': os.path.relpath(part_graph_file, out_path)}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)
        save_graphs(part_graph_file, [part])
        
    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)
    print('Save partitions: {:.3f} seconds'.format(time.time() - start))
    if return_mapping:
        return orig_nids, orig_eids
    # new_sampling_function.sampling_function(g, 5, 2, partition_sets, xadj, adjncy, sample_count=5)
