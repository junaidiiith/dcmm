import time
from gnn_models.custom_gnn import GraphClusteringLoss
from tqdm.auto import tqdm
import torch
from gnn_models.dmon import Single
from trainers.utils import device, get_modularization_scores, get_output_dim
from trainers.utils import (
    get_nx_adj, 
    get_modularization_scores
)

import settings


DMON_EPOCHS = 2000
DMON_LR = 0.001


def run(g, X):
    print("Device: ", device)
    A = get_nx_adj(g)
    ips = (X.unsqueeze(0).to(device), A.unsqueeze(0).to(device))
    model = Single(
        X.shape[1], 
        get_output_dim(g),
        skip_conn=False, 
        collapse_regularization=0.1,
        # gl=GraphClusteringLoss()
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DMON_LR)

    model.train()
    start_time = time.time()
    all_metrics = list()
    # for epoch in range(DMON_EPOCHS):
    for epoch in tqdm(range(DMON_EPOCHS), desc='DMoN Epochs'):
        optimizer.zero_grad()
        _, pred, _, losses = model(ips)
        loss = torch.FloatTensor([0]).to(device)
        for loss_val in losses.values():
            if loss_val is not None:
                loss += loss_val

        loss.backward()
        optimizer.step()
        clusters = {node: cluster for node, cluster in zip(
        g.nodes(), pred[0].detach().cpu().numpy().argmax(axis=-1))}
        
        metrics = get_modularization_scores(g, clusters)
        metrics['clusters'] = clusters
        metrics['epoch'] = epoch
        metrics['time'] = time.time() - start_time
        all_metrics.append(metrics)

        if epoch % 50 == 0 and settings.verbose:
            print(f'Epoch: {epoch}, Loss: {loss.item()}, Cohesion: {metrics["cohesion"]}, Coupling: {metrics["coupling"]}')

    max_cohesion_result = max(all_metrics, key=lambda x: x['cohesion'])
    min_coupling_result = min(all_metrics, key=lambda x: x['coupling'])


    result = {
        'max_cohesion': max_cohesion_result,
        'min_coupling': min_coupling_result,
        'total_time': time.time() - start_time,
    }

    return result