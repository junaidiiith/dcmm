import networkx as nx
from scipy.sparse.linalg import eigsh
import settings
import sys
import torch
from torch_geometric.nn import Node2Vec
from tqdm.auto import tqdm
from trainers.utils import device
import trainers.utils as utils


NODE2VEC_EPOCHS = 50
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_CONTEXT_SIZE = 20
NODE2VEC_DIM = 64
NODE2VEC_NEG_SAMPLES = 4
NODE2VEC_BATCH_SIZE = 128
NODE2VEC_LR = 0.01
NODE2VEC_WALKS_PER_NODE = 30
NODE2VEC_NUM_WORKERS = 4 if sys.platform == 'linux' else 0
NODE2VEC_P = 1
NODE2VEC_Q = 1


def get_node2vec_embedding(g):
    edge_index = utils.get_edge_index(g)
    node2vec = Node2Vec(
        edge_index,
        embedding_dim=NODE2VEC_DIM,
        walk_length=NODE2VEC_WALK_LENGTH,
        context_size=NODE2VEC_CONTEXT_SIZE,
        walks_per_node=NODE2VEC_WALKS_PER_NODE,
        num_negative_samples=NODE2VEC_NEG_SAMPLES,
        p=NODE2VEC_P,
        q=NODE2VEC_Q,
        sparse=True,
    ).to(device)

    num_workers = NODE2VEC_NUM_WORKERS
    loader = node2vec.loader(batch_size=NODE2VEC_BATCH_SIZE, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=NODE2VEC_LR)
    node2vec.train()
    total_loss = 0
    for epoch in range(1, NODE2VEC_EPOCHS + 1):
    # for epoch in tqdm(range(1, NODE2VEC_EPOCHS + 1), desc='Training Node2Vec For Node Embeddings'):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        loss = total_loss / len(loader)
        if epoch % 20 == 0 and settings.verbose:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    return node2vec.embedding.weight.detach().cpu().numpy()


def laplacian_eigenmaps_embeddings(G, dimensions=NODE2VEC_DIM):
    dimensions = min(dimensions, G.number_of_nodes() - 2)
    L = nx.laplacian_matrix(G).astype(float)
    _, eigvecs = eigsh(L, k=dimensions+1, which='SM')
    embeddings = eigvecs[:, 1:]  # Skip the first eigenvector
    return embeddings


def get_adjacency_matrix_embedding(G):
    A = nx.adjacency_matrix(G).todense()
    return A