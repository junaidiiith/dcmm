# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Layers for the graph neural networks."""

from collections import OrderedDict
import math
from typing import Callable, Dict, Tuple, Union

import torch as T
from torch import nn
from torch.nn import functional as F

from gnn_models.dmon_utils import normalize_graph


device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
  """Graph Convolutional Network Layer.

  This implementation is based on the method prposed
  in https://arxiv.org/abs/1609.02907 by Kipf, T.N., et al.
  and the idea in https://arxiv.org/pdf/2006.16904.pdf.
  by Tsitsulin, A., et al.

  Attributes
    in_dim: dimension of input node embedding.
    out_dim: dimension of output node embedding.
  """

  __slots__ = ['in_dim', 'out_dim', 'conv_w',
               'skip_w', 'bias', 'activation']
  def __init__(self, in_dim: int, out_dim: int,
                bias: bool=False, skip_conn: bool=True,
                activation: Union[Callable, str]='selu') -> None:
    """Initialize single layer of Graph Convolutional Network

    Args:
      in_dim: An integer dimension of input node embedding.
      out_dim: An integer dimension of output node embedding.
      bias: If include bias in the convolution computation.
      skip_conn: If use skip connection.
      activation: Activation function.
    """
    super().__init__()

    assert isinstance(in_dim, int),\
      TypeError(f'Expect int dimension, but got {type(in_dim):s}.')
    assert isinstance(out_dim, int),\
      TypeError(f'Expect int dimension, but got {type(out_dim):s}.')

    self.in_dim = in_dim
    self.out_dim = out_dim
    if isinstance(activation, Callable):
      self.activation = activation
    elif isinstance(activation, str):
      if activation == 'relu':
        self.activation = F.relu
      elif activation == 'selu':
        self.activation = F.selu
      else:
        self.activation = F.silu
    else:
      raise ValueError('GCN activation of unknown type!')

    self.conv_w = nn.Parameter(T.FloatTensor(in_dim, out_dim)).to(device)
    # Residual connection in GCN
    if skip_conn:
      self.skip_w = nn.Parameter(T.FloatTensor(out_dim)).to(device)
    else:
      self.register_parameter('skip_w', None)

    # Bias
    if bias:
      self.bias = nn.Parameter(T.FloatTensor(out_dim)).to(device)
    else:
      self.register_parameter('bias', None)

    self.init_parameters()

  def __repr__(self):
    """Name of the layer."""
    return self.__class__.__name__ +\
            f'({self.in_dim:d}->{self.out_dim:d})'

  def init_parameters(self):
    """Initialize model parameters."""
    stdv = 1. / math.sqrt(self.out_dim)
    self.conv_w.data.uniform_(-stdv, stdv)
    if self.skip_w is not None:
      self.skip_w.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)


  def forward(self, inputs: Tuple[T.Tensor]) -> T.Tensor:
    """Forward function for naive graph convolution.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[N, N]` square adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is a `[N, d]` node
      embedding updated by graph convolution and the other original
      square matrix.
    """

    features, graph = inputs
    features = features.float().to(device)
    graph = graph.float().to(device)
    output = T.matmul(features, self.conv_w)

    # Skip connection
    if self.skip_w is not None:
      output = output * self.skip_w + T.matmul(graph, output)
    else:
      output = T.matmul(graph, output)

    # Additive bias
    if self.bias is not None:
      output = output + self.bias

    # Nonlinear
    output = self.activation(output)

    return output, graph

class DMoN(nn.Module):
  """PyTorch re-implementation of Deep Modularity Network (DMoN).

  Attributes:
    n_clusters: Number of clusters in the model.
    collapse_regularization: Weight for collapse regularization.
    do_unpooling: If perform unpooling of the features with respect to
    their soft clusters. If true, shape of the input is preserved.
  """

  def __init__(self, in_dim: int, n_clusters: int, dropout: float=0.,
               activation: Union[str, Callable]='selu',
               collapse_regularization: float=0.1,
               do_unpooling: bool=False, gl=None) -> None:
    """Initialize the Deep Modularity Network.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      dropout: A float dropout probability of encoder.
      activation: Activation function.
      collapse_regularization: A float weight for regularization.
      do_unpooling: If perform unpooling of the feature.
    """

    super().__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.do_unpooling = do_unpooling
    if isinstance(activation, Callable):
      self.activation = activation
    elif isinstance(activation, str):
      if activation == 'relu':
        self.activation = F.relu
      elif activation == 'selu':
        self.activation = F.selu
      else:
        self.activation = F.silu
    else:
      raise ValueError('GCN activation of unknown type!')

    self.transform = nn.Sequential(nn.Linear(in_dim, n_clusters), nn.Dropout(p=dropout))
    self.gl = gl


    self.init_parameters()

  def init_parameters(self):
    """Initialize model parameters."""

    nn.init.orthogonal_(self.transform[0].weight)
    nn.init.zeros_(self.transform[0].bias)

  def forward(self, inputs: Tuple[T.Tensor, T.Tensor])\
      -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
    """Perform DMoN clustering according to node features and graph.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[B, N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[B, N, N]` square adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is a `[B, k, d]` cluster
      representations with `k` as the number of clusters. The second tensor
      is a `[B, N, k]` cluster assignment matrix. If do_unpooling is True,
      return `[B, N, d]` node representations instead of cluster representation.
    """
    
    nodes, adjacency = inputs
    batch_size = nodes.size(0)
    num_nodes = nodes.size(1)
    if not nodes.shape[0] == adjacency.shape[0]:
      adjacency = adjacency.repeat(batch_size, 1, 1)
    
    nodes, adjacency = nodes.to(device), adjacency.to(device)

    assert isinstance(nodes, T.Tensor) and isinstance(adjacency, T.Tensor),\
      TypeError(f'Expect Tensors, but got {type(nodes)} and {type(adjacency)}.')
    assert adjacency.shape[1] == nodes.shape[1],\
      ValueError('Node number in adjacency matrix does not match feature.')
    assert adjacency.shape[1] == adjacency.shape[2],\
      ValueError(f'Expect square adjacency matrix, but got {adjacency.shape}.')

    # Compute soft cluster assignments with normalized adjacency matrix
    # print("NodeS devICE", nodes.device)
    assignments = F.softmax(self.transform(nodes.to(device)), dim=-1)
    cluster_sizes = assignments.sum(dim=1)  # number of nodes in each cluster
    assignments_pooling = assignments / cluster_sizes.unsqueeze(1)

    degrees = T.sum(adjacency, dim=1).unsqueeze(-1)  # shape: [B, N, 1]
    num_edges = degrees.sum(dim=[-1, -2]) # shape: [B, ]

    # Calculate the pooled graph C^T*A*C of shape [B, k, k]
    pooled_graph = T.matmul(adjacency, assignments).permute(0, 2, 1)
    pooled_graph = T.matmul(pooled_graph, assignments)

    # Calculate the dyad normalizer matrix C^T*d^T*d*S of shape [B, k, k]
    dyad_left = T.matmul(assignments.permute(0, 2, 1), degrees)
    dyad_right = T.matmul(degrees.permute(0, 2, 1), assignments)
    normalizer = T.matmul(dyad_left, dyad_right)
    normalizer = normalizer / 2 / num_edges[:, None, None]
 
    # Calculate deep modularity loss
    
    modularity_loss = -T.diagonal(pooled_graph-normalizer, dim1=-2, dim2=-1)\
                        .sum() / 2 / num_edges / batch_size
    modularity_loss = T.mean(modularity_loss, dim=0)
  

    # Calculate collapse regularization
    
    collapse_loss = T.norm(cluster_sizes, dim=-1) / num_nodes\
                    * T.sqrt(T.FloatTensor([self.n_clusters])).to(device) - 1
    collapse_loss: T.Tensor = self.collapse_regularization * collapse_loss
    collapse_loss = T.mean(collapse_loss, dim=0)  # Batch mean
    

    # Calcualte pooled features
    pooled_features = T.matmul(assignments_pooling.permute(0, 2, 1), nodes)
    # Nonlinear
    pooled_features = self.activation(pooled_features)

    # Unpooling
    if self.do_unpooling:
      pooled_features = T.matmul(assignments_pooling, pooled_features)

    if self.gl is not None:
      gl_loss = self.gl(adjacency.squeeze(0), assignments.squeeze(0))
    
    all_losses = {
      'modularity': modularity_loss, 
      'collapse': collapse_loss, 
      'gl': gl_loss if self.gl is not None else None
    }

    return pooled_features, assignments, all_losses


class Single(nn.Module):
  """
  This is a graph neural network model for airport-based operational delay
  clustering. It's a single-level graph clustering model with each node
  associated to a local airport and edges as their connectivity.

  Attributes:
    in_dim: Dimension of input node embeddings.
  """

  __slots__ = ['in_dim', 'encoder', 'predict', 'dropout']
  def __init__(self, in_dim: int, n_clusters: int,
               hidden: int=128, depths: int=1, dropout:float=0.,
               inflation: int=1, activation: str='silu', skip_conn: bool=True,
               collapse_regularization: float=0.1,
               do_unpooling: bool=False, gl=None) -> None:
    """Initialize GNN clustering model.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      hidden: Number of neurons in the hidden layers.
      depths: Number of hidden layers.
      dropout: A float probability of dropout layer.
      inflation: Inflation factor between layers.
      activation: Name of activation function to use.
      skip_conn: If use skip connected graph layers.
      collapse_regularization: A float weight for regularization.
      do_unpooling: If perform unpooling of the feature.
    """

    super().__init__()
    self.in_dim = in_dim
    self.skip_conn = skip_conn

    # Build model
    self.register_parameter('glnn', None)

    layers = OrderedDict()
    emb_dim = in_dim
    for i in range(depths):
      layers[f'gcn_{i:d}'] = GCNLayer(emb_dim, hidden,
                                      skip_conn=skip_conn,
                                      activation=activation)
      emb_dim = hidden
      hidden = hidden * inflation
    self.encoder = nn.Sequential(layers)
    self.dmon = DMoN(hidden, n_clusters, dropout,
                     activation=activation,
                     collapse_regularization=collapse_regularization,
                     do_unpooling=do_unpooling, gl=gl)

  def forward(self, inputs: Tuple[T.Tensor, T.Tensor])\
      -> Tuple[T.Tensor, T.Tensor, T.Tensor, Dict[str, T.Tensor]]:
    """Forward function for single level.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is an optional `[N, N]` target adjacency matrix.

    Returns:
      A tuple of PyTorch tensors and a dictionary. The first tensor is a
      `[B, k, d]` centroid embedding of `k` cluster centroids. If do_unpooling,
      it becomes a `[B, N, d]` unpooled feature tensor instead. The second
      tensor is a `[B, N, k]` cluster label tensor. The third tensor is a
      `[B, N, N]` adjacency matrix tensor. The last element is a dictionary
      consisting of all the relevant losses.
    """

    assert len(inputs) == 2,\
      ValueError(f'Expect input to have 2 elements, but got {len(inputs)}.')

    features, graph = inputs
    features = features.float()
    graph = graph.float()
    if graph is not None:
      graph = graph.float()
      if self.skip_conn:
        norm_graph = normalize_graph(graph.clone(), add_self_loops=False)
      else:
        norm_graph = normalize_graph(graph.clone(), add_self_loops=True)
    else:
      norm_graph = None

    features, _ = self.encoder([features, norm_graph])  # norm graph for gcn

    features = features.float().to(device)
    pooled_features, pred, losses = self.dmon.forward([features, graph])


    return pooled_features, pred, norm_graph, losses
