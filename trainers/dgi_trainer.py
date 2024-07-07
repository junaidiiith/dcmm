import time
import torch
from gnn_models.DGI import DeepGraphInfomax, Encoder, Summarizer, corruption, cluster_net

from tqdm.auto import tqdm
from trainers.utils import (
    device, 
    get_modularization_scores, 
    get_nx_adj,
    get_edge_index,
    get_output_dim
)

import settings

DGI_EPOCHS = 2000
DGI_LR = 0.001
DGI_HIDDEN_DIM = 128

def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]).to(device) - torch.eye(adj.shape[0]).to(device))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod


def run(g, X):
    print("Device: ", device)
    edge_index = get_edge_index(g)
    adj = get_nx_adj(g)
     
    model = DeepGraphInfomax(
        hidden_channels=DGI_HIDDEN_DIM, 
        encoder=Encoder(X.shape[1], DGI_HIDDEN_DIM),
        out_channels=get_output_dim(g),
        summary=Summarizer(),
        corruption=corruption,
        cluster=cluster_net)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DGI_LR, weight_decay=5e-3)

    adj = adj.float().to(device)
    mod = make_modularity_matrix(adj)
    model.train()
    epoch_losses = list()
    all_metrics = list()

    start_time = time.time()
    for epoch in tqdm(range(DGI_EPOCHS), desc='DGI Epochs'):
    # for epoch in range(DGI_EPOCHS):
        optimizer.zero_grad()
        pos_z, neg_z, summary, mu, r, _ = model(X.to(device), edge_index.to(device))
        dgi_loss = model.loss(pos_z, neg_z, summary)
        modularity_loss = model.modularity(r, adj, mod)
        comm_loss = model.comm_loss(pos_z, mu)
        # loss = -modularity_loss
        loss = 5*dgi_loss - modularity_loss + comm_loss

        # print(f"Epoch: {epoch}, Loss: {loss.item()}")
        epoch_losses.append(loss.item())
        
        clusters = {node: cluster for node, cluster in zip(
        g.nodes(), r.detach().cpu().numpy().argmax(axis=-1))}

        metrics = get_modularization_scores(g, clusters)
        metrics['clusters'] = clusters
        metrics['epoch'] = epoch
        ## time in seconds
        metrics['time'] = time.time() - start_time
        all_metrics.append(metrics)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 and settings.verbose:
            print(f'Epoch: {_}, Loss: {loss.item()}, Cohesion: {metrics["cohesion"]}, Coupling: {metrics["coupling"]}')

    

    max_cohesion_result = max(all_metrics, key=lambda x: x['cohesion'])
    min_coupling_result = min(all_metrics, key=lambda x: x['coupling'])

    result = {
        'max_cohesion': max_cohesion_result,
        'min_coupling': min_coupling_result,
        'total_time': time.time() - start_time
    }

    return result