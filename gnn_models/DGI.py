import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn.inits import reset, uniform
import sklearn
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

EPS = 1e-15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, out_channels, summary, corruption, cluster):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()
        self.K = out_channels
        self.cluster_temp = 30
        self.init = torch.rand(self.K, hidden_channels)
        self.cluster = cluster
        
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)


    def forward(self, x, edge_index):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(x, edge_index)
        cor = self.corruption(x, edge_index)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z)
        num_iter = 1
        mu_init, _, _ = self.cluster(pos_z, self.K, num_iter, self.cluster_temp, self.init)
        mu, r, dist = self.cluster(pos_z, self.K, 1, self.cluster_temp, mu_init.detach().clone())
        return pos_z, neg_z, summary, mu, r, dist

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        #print("shape", z.shape,summary.shape)
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        # print('pos_loss = {}, neg_loss = {}'.format(pos_loss, neg_loss))
        # bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        # modularity = (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
        return pos_loss + neg_loss #+ modularity

    def comm_loss(self, pos_z, mu):
        return -torch.log(self.discriminate(pos_z, self.summary(mu), sigmoid=True) + EPS).mean()

    def modularity(self, r, bin_adj, mod):
        r, bin_adj, mod = r.to(device), bin_adj.to(device), mod.to(device)
        bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]).to(device) - torch.eye(bin_adj.shape[0]).to(device))
        return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
    
    def spectral_clustering_loss(self, X, y, alpha):
        n_clusters = y.shape[1]
        norm = torch.norm(X, p=2, dim=1, keepdim=True)
        X = X / norm

        # Compute affinity matrix A using Gaussian kernel
        A = torch.exp(-torch.norm(X[:, None, :] - X[None, :, :], dim=-1) ** 2)

        # Compute degree matrix D
        D = torch.diag(torch.sum(A, dim=1))

        # Compute Laplacian matrix L
        L = D - A

        # Compute eigenvectors of L corresponding to the smallest eigenvalues
        _, eigvecs = torch.linalg.eigh(L, UPLO='U')

        # Normalize the eigenvectors
        eigvecs = F.normalize(eigvecs[:, :n_clusters], dim=1)

        # Compute cluster assignments
        y_pred = F.softmax(torch.mm(eigvecs, eigvecs.t()), dim=1)

        # Compute spectral clustering loss
        loss = -torch.trace(torch.mm(y.t(), torch.log(y_pred))) + alpha * torch.trace(torch.mm(y.t(), L))

        return loss

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return f'{self.__class__.__name__}(hidden_dim={self.hidden_channels}, encoder={self.encoder},\
              summary={self.summary}, weight={self.weight}, K={self.K}, cluster={self.cluster})'



def GELU(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels) # , cached=True)
        # self.gat = GATConv(in_channels, 64, heads=8, dropout=0.0)
        self.prelu = nn.PReLU(hidden_channels)
        # self.ac = nn.ELU()
        # self.prop = APPNP(10, 0.1)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        # x = self.prop(x, edge_index)
        return x
    
        

class Summarizer(nn.Module):
    def __init__(self):
        super(Summarizer, self).__init__()
    
    def forward(self, z):
        return torch.sigmoid(z.mean(dim=0))

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def cluster_net(data, k, num_iter, cluster_temp, init):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np**2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0: return init
    mu = init
    mu = mu.to(device)
    
    data = data.to(device)
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for _ in range(num_iter):
        #get distances between all data points and cluster centers
#        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1).to(device)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist

'''
def summary(z, x, edge_index):
    capsule_model = CapsuleLayer(z.size(1), z.size(1))
    comm_emb = capsule_model(z.unsqueeze(0)).squeeze(0)
    return torch.sigmoid(comm_emb.mean(dim=0))
'''
