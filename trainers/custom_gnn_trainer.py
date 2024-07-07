import time
import torch
from tqdm.auto import tqdm
from gnn_models.custom_gnn import GNNModel, GraphClusteringLoss
from embedding import NODE2VEC_DIM
from trainers.utils import (
    device, 
    get_edge_index, 
    get_modularization_scores, 
    get_nx_adj,
    get_output_dim
)

import settings


GNN_NUM_EPOCHS = 2000
GNN_MODEL_NAME = 'SAGEConv'
GNN_INPUT_DIM = NODE2VEC_DIM
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_RESIDUAL = True
GNN_LNORM = True
GNN_DROPOUT = 0.1
GNN_LR = 0.001



def run(g, X):
    print("Device: ", device)
    loss_fn = GraphClusteringLoss()
    
    edge_index = get_edge_index(g)
    A = get_nx_adj(g).to(device)
    
    model = GNNModel(
        model_name=GNN_MODEL_NAME, 
        input_dim=X.shape[1], 
        hidden_dim=GNN_HIDDEN_DIM, 
        out_dim=get_output_dim(g), 
        num_layers=GNN_NUM_LAYERS, 
        residual=GNN_RESIDUAL, 
        l_norm=GNN_LNORM, 
        dropout=GNN_DROPOUT
    ).to(device)

    ## loss_fn.lambda_param + model.parameters
    train_params = list(model.parameters()) + [loss_fn.lambda_param]

    model.train()
    optimizer = torch.optim.Adam(train_params, lr=GNN_LR)
    all_metrics = list()
    start_time = time.time()
    for epoch in tqdm(range(1, GNN_NUM_EPOCHS + 1), desc='GNN Epochs'):
    # for epoch in range(1, GNN_NUM_EPOCHS + 1):
        optimizer.zero_grad()
        Y = model(X.to(device), edge_index.to(device))
        loss = loss_fn(A, Y)
        loss.backward()
        optimizer.step()
        node_clusters = torch.argmax(Y, dim=1).cpu()
        clusters = {i: c.item() for i, c in enumerate(node_clusters)}
        metrics = get_modularization_scores(g, clusters)
        metrics['clusters'] = clusters
        metrics['time'] = time.time() - start_time
        metrics['epoch'] = epoch
        all_metrics.append(metrics)

        if epoch % 50 == 0 and settings.verbose:
            print(f'Epoch: {epoch}, Loss: {loss.item()}, Cohesion: {metrics["cohesion"]}, Coupling: {metrics["coupling"]}')

    
    max_cohesion_result = max(all_metrics, key=lambda x: x['cohesion'])
    min_coupling_result = min(all_metrics, key=lambda x: x['coupling'])


    result = {
        'max_cohesion': max_cohesion_result,
        'min_coupling': min_coupling_result,
        'total_time': time.time() - start_time
    }

    return result