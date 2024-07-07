import argparse
from collections import defaultdict
import pickle
import random
import time
import networkx as nx
import numpy as np
import pandas as pd
import torch
from trainers import (
    custom_gnn_trainer, 
    dmon_trainer, 
    dgi_trainer
)
from tqdm.auto import tqdm


from embedding import (
    get_adjacency_matrix_embedding, 
    laplacian_eigenmaps_embeddings,
    get_node2vec_embedding
)

import settings

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_pickle_path', type=str, default='dataset/ecore_non_dup_models.pkl')
    args.add_argument('--results_dir', type=str, default='results')
    args.add_argument('--min_nodes', type=int, default=-1)
    args.add_argument('--max_nodes', type=int, default=-1)
    args.add_argument('--seed', type=int, default=1331)
    args.add_argument('--runs', type=int, default=1)

    args.add_argument('--embedding', type=str, choices=['node2vec', 'laplacian', 'adj'], default='node2vec')

    args.add_argument('--dgi', action='store_true')
    args.add_argument('--dmon', action='store_true')
    args.add_argument('--gnn', action='store_true')
    args.add_argument('--all', action='store_true')

    args.add_argument('--verbose', action='store_true')
    return args.parse_args()


def print_result(key, result):
    print(f"{key} Results")
    for k, v in result.items():
        if isinstance(v, float):
            print(f'{k} --> {v}')
        else:
            print(f'{k} --> cohesion: {v["cohesion"]}, coupling: {v["coupling"]}, clusters: {len(set(v["clusters"].values()))}')
    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    data_pickle_path = args.data_pickle_path
    results_dir = args.results_dir
    settings.verbose = args.verbose


    with open(data_pickle_path, 'rb') as f:
        duplicate_models = pickle.load(f)

    duplicate_numbered_graphs = [
        (a, nx.convert_node_labels_to_integers(b)) for a, b in duplicate_models
        if list(nx.isolates(b)) == [] and (args.min_nodes == -1 or len(b.nodes) >= args.min_nodes)
        and (args.max_nodes == -1 or len(b.nodes) <= args.max_nodes)
    ]
    print(f'Number of graphs: {len(duplicate_numbered_graphs)}')

    assert args.gnn or args.dgi or args.dmon or args.all, 'At least one of the following must be set: --dgi, --dmon, --gnn or --all'

    results_str = ''
    results_str += 'dgi_' if args.dgi else '' 
    results_str += 'gnn_' if args.gnn else '' 
    results_str += 'dmon_' if args.dmon else ''
    results_str += 'all_' if args.all else ''
    results_str += f'_max_{args.max_nodes}_min_{args.min_nodes}'
    results_path = f'{results_dir}/{results_str}_results.xlsx'


    metrics_results = defaultdict(list)
    rows = list()

    for run in range(args.runs):
        print(f'Run {run}')
        for i, (file_name, graph) in tqdm(enumerate(duplicate_numbered_graphs), total=len(duplicate_numbered_graphs), desc='Graphs'):
            print(f'{file_name} - {len(graph.nodes)} nodes, {len(graph.edges)} edges')
            metrics = dict()
            embed_start_time = time.time()
            if args.embedding == 'node2vec':
                X = get_node2vec_embedding(graph)
            elif args.embedding == 'laplacian':
                X = laplacian_eigenmaps_embeddings(graph)
            elif args.embedding == 'adj':
                X = get_adjacency_matrix_embedding(graph)
            
            embed_time = time.time() - embed_start_time
            print(f'Embedding time: {embed_time}')
            X = torch.tensor(X, dtype=torch.float32)

            if args.dgi or args.all:
                metrics['dgi'] = dgi_trainer.run(graph, X)
                # print_result('DGI', metrics['dgi'])
            
            if args.gnn or args.all:
                metrics['gnn'] = custom_gnn_trainer.run(graph, X)
                print_result('GNN', metrics['gnn'])
            
            if args.dmon or args.all:
                metrics['dmon'] = dmon_trainer.run(graph, X)
                # print_result('DMoN', metrics['dmon'])
                

            num_nodes = len(graph.nodes)
            num_edges = len(graph.edges)
            edge_nodes_ratio = num_edges / num_nodes

            metrics_results[file_name].append(metrics)
            rows.append({
                'file_name': file_name,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'edge_nodes_ratio': edge_nodes_ratio,
                'metrics': metrics_results[file_name],
                'embed_time': embed_time
            })
            
        
        pd.DataFrame(rows).to_excel(f'{results_path}', index=False)