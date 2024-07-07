import torch
import networkx as nx

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NODES_PER_CLUSTER = 7

prefix = '../datasets/ModelClassification/modelset/raw-data/repo-ecore-all/data/'

def get_file_name(f):
    return f.replace(prefix, '').replace('.ecore', '').replace('/', '_')

def get_edge_index(nxg):
    edge_index = torch.tensor(list(nxg.edges)).t().contiguous()
    return edge_index


def get_nx_adj(nxg) -> torch.Tensor:
    adj = nx.adjacency_matrix(nxg).todense()
    return torch.tensor(adj, dtype=torch.long)


def get_edges_in_undirected_graph(graph):
    edges = set()
    for edge in graph.edges():
        if (edge[0] != edge[1]) and (edge[0], edge[1]) not in edges and (edge[1], edge[0]) not in edges:
            edges.add(edge)
    
    return edges


def remove_isolated_nodes(graph):
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    return graph


def get_output_dim(graph):
    return max(10, graph.number_of_nodes() // NODES_PER_CLUSTER)


def get_coupling(graph, partition):
    clusters = list(set(partition.values()))
    edges_between_clusters = list()
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            cluster1, cluster2 = clusters[i], clusters[j]
            sub_graph1 = graph.subgraph([node for node in partition if partition[node] == cluster1])
            sub_graph2 = graph.subgraph([node for node in partition if partition[node] == cluster2])
            edges_between_clusters += list(nx.edge_boundary(graph, sub_graph1, sub_graph2))

    coupling = len(edges_between_clusters) /  graph.number_of_edges()

    if coupling > 1:
        print('Coupling > 1')
        print(graph.number_of_edges())
        print(coupling)
        exit(0)

    return coupling



def get_cohesion(graph, partition):
    clusters = set(partition.values())
    # print(len(clusters))
    cohesion = 0
    for cluster in clusters:
        sub_graph = graph.subgraph([node for node in partition if partition[node] == cluster])
        max_edges = sub_graph.number_of_nodes() * (sub_graph.number_of_nodes() - 1) / 2
        edges = get_edges_in_undirected_graph(sub_graph)
        cohesion += len(edges) / max_edges if max_edges != 0 else 0

    cohesion /= len(clusters)
    return cohesion


def get_modularization_scores(graph, partition):
    scores = {
        'cohesion': get_cohesion(graph, partition),
        'coupling': get_coupling(graph, partition),
    }
    return scores