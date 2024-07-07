import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F



class GNNModel(torch.nn.Module):
    """GNN Model with multiple layers"""
    def __init__(self, model_name, input_dim, hidden_dim, out_dim, num_layers, num_heads=None, residual=False, l_norm=False, dropout=0.1):
        super(GNNModel, self).__init__()
        gnn_model = getattr(torch_geometric.nn, model_name)
        self.conv_layers = nn.ModuleList()
        if model_name == 'GINConv':
            input_layer = gnn_model(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()), train_eps=True)
        elif num_heads is None:
            input_layer = gnn_model(input_dim, hidden_dim, aggr='SumAggregation')
        else:
            input_layer = gnn_model(input_dim, hidden_dim, heads=num_heads, aggr='SumAggregation')
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            if model_name == 'GINConv':
                self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), train_eps=True))
            elif num_heads is None:
                self.conv_layers.append(gnn_model(hidden_dim, hidden_dim, aggr='SumAggregation'))
            else:
                self.conv_layers.append(gnn_model(num_heads*hidden_dim, hidden_dim, heads=num_heads, aggr='SumAggregation'))

        if model_name == 'GINConv':
            self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU()), train_eps=True))
        else:
            self.conv_layers.append(gnn_model(hidden_dim if num_heads is None else num_heads*hidden_dim, out_dim, aggr='SumAggregation'))
            
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim if num_heads is None else num_heads*hidden_dim) if l_norm else None
        self.residual = residual
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_feat, edge_index):
        h = in_feat
        h = self.conv_layers[0](h, edge_index)
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index) if not self.residual else conv(h, edge_index) + h
            h = self.activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index)
        return h


class GraphClusteringLoss(nn.Module):
    def __init__(self):
        super(GraphClusteringLoss, self).__init__()
        # Initialize lambda as a learnable parameter
        self.lambda_param = nn.Parameter(torch.tensor(0.5))  # Initial value of lambda


    def forward(self, A: torch.Tensor, Y: torch.Tensor):
        n, C = Y.shape

        Y = F.softmax(Y, dim=1)  # Shape: n x C

        Z = Y.T.unsqueeze(2) * Y.T.unsqueeze(1)
        total_ewc = (Z * A.unsqueeze(0)).sum(dim=(1, 2))
        nwc = Y.sum(dim=0)  # Shape: C
        max_ewc = (nwc * (nwc - 1) + 1e-8) / 2  # Avoid division by zero

        cohesion = (total_ewc / max_ewc).sum()

        inter_cluster_edges = 0
        for i in range(C):
            for j in range(i+1, C):
                Y_i = Y[:, i].unsqueeze(1)
                Y_j = Y[:, j].unsqueeze(1)
                A_ij = Y_i @ Y_j.T
                ice = (A_ij * A).sum()
                inter_cluster_edges += ice
            
        total_edges = A.sum()
        coupling = inter_cluster_edges / (total_edges + 1e-9)  # Avoid division by zero

        loss = -cohesion + coupling

        # node_clusters = torch.argmax(Y, dim=1).cpu().numpy()
        # clusters = {i: c.item() for i, c in enumerate(node_clusters)}
        # g = nx.from_numpy_array(A.detach().cpu().numpy())
        # metrics = get_modularization_scores(g, clusters)

        # if settings.verbose:
        #     print(f'Loss: {loss.item():.4f}, Cohesion: {cohesion.item():.4f}, Coupling: {coupling.item():.4f}, Lambda: {self.lambda_param.item():.4f}')
        #     print(f'Actual Cohesion: {metrics["cohesion"]:.4f}, Actual Coupling: {metrics["coupling"]:.4f}, Clusters: {len(set(clusters.values()))}')

        return loss
    