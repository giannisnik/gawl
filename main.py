import argparse
import networkx as nx
import numpy as np

from math import sqrt
from evaluate import eval_10_fold

# Argument parser
parser = argparse.ArgumentParser(description='GAWL')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--T', type=int, default=4, help='Iterations of WL algorithm')
args = parser.parse_args()


def load_data(ds_name, use_node_labels):
    node2graph = {}
    Gs = []

    with open("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open("datasets/%s/%s_A.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), "r") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c]-1].nodes[c]['label'] = node_label
                c += 1

    labels = []
    with open("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    labels  = np.array(labels, dtype=np.float32)
    return Gs, labels


def get_wl_labels(Gs, h, node_labels_exist):
    N = len(Gs)

    node_labels = list()
    node_to_idx = list()
    for i,G in enumerate(Gs):
        node_labels.append(np.zeros((G.number_of_nodes(), h+1), dtype=np.int64))
        node_to_idx.append(dict())
        for j,node in enumerate(G.nodes()):
            node_to_idx[i][node] = j

            if not node_labels_exist:
                G.nodes[node]['label'] = G.degree(node)

    max_label_counts = list()

    label_to_idx = dict()
    for i,G in enumerate(Gs):
        for j,node in enumerate(G.nodes()):
            if G.nodes[node]['label'] not in label_to_idx:
                label_to_idx[G.nodes[node]['label']] = len(label_to_idx)

            node_labels[i][j,0] = label_to_idx[G.nodes[node]['label']]

    label_counts = np.zeros(len(label_to_idx), dtype=np.int32)
    for i in range(len(node_labels)):
        unique, counts = np.unique(node_labels[i][:,0], return_counts=True)
        v = np.zeros(len(label_to_idx), dtype=np.int32)
        for j in range(unique.size):
            v[unique[j]] = counts[j]
        label_counts = np.maximum(label_counts, v)

    max_label_counts.append(label_counts)

    for it in range(1,h+1):
        label_to_idx = dict()
        for i,G in enumerate(Gs):
            for j,node in enumerate(G.nodes()):
                new_label = list()
                for neighbor in G.neighbors(node):
                    new_label.append(node_labels[i][node_to_idx[i][neighbor],it-1])
                new_label = sorted(new_label)
                new_label.insert(0, node_labels[i][j,it-1])
                new_label = tuple(new_label)
                if new_label not in label_to_idx:
                    label_to_idx[new_label] = len(label_to_idx)

                node_labels[i][j,it] = label_to_idx[new_label]

        label_counts = np.zeros(len(label_to_idx), dtype=np.int32)
        for i in range(len(node_labels)):
            unique, counts = np.unique(node_labels[i][:,it], return_counts=True)
            v = np.zeros(len(label_to_idx), dtype=np.int32)
            for j in range(unique.size):
                v[unique[j]] = counts[j]
            label_counts = np.maximum(label_counts, v)

        max_label_counts.append(label_counts)

    return node_labels, max_label_counts


def compute_gawl_kernel(Gs, h, node_labels, max_label_counts):
    K = np.zeros((len(Gs), len(Gs)))
    for it in range(h+1):
        edge_to_idx = dict()
        edges = list()
        node_label_freq = list()
        for i,G in enumerate(Gs):
            node_to_idx = dict()
            for j,node in enumerate(G.nodes()):
                node_to_idx[node] = j

            unique, counts = np.unique(node_labels[i][:,it], return_counts=True)
            node_label_freq.append(dict())
            for j in range(unique.size):
                node_label_freq[i][unique[j]] = counts[j]

            edges.append(dict())
            for edge in G.edges():
                v1 = node_labels[i][node_to_idx[edge[0]],it]
                v2 = node_labels[i][node_to_idx[edge[1]],it]
                if v1 < v2:
                    if (v1, v2) not in edge_to_idx:
                        edge_to_idx[(v1, v2)] = len(edge_to_idx)
                    if (v1, v2) in edges[i]:
                        edges[i][(v1, v2)] += 1
                    else:
                        edges[i][(v1, v2)] = 1
                else:
                    if (v2, v1) not in edge_to_idx:
                        edge_to_idx[(v2, v1)] = len(edge_to_idx)
                    if (v2, v1) in edges[i]:
                        edges[i][(v2, v1)] += 1
                    else:
                        edges[i][(v2, v1)] = 1

        for i in range(len(Gs)):
            for j in range(i,len(Gs)):
                for edge in edges[i]:
                    if edge in edges[j]:
                        if edge[0] == edge[1] and node_label_freq[i][edge[0]] > 1 and node_label_freq[j][edge[0]] > 1:
                            ei = 2*edges[i][edge]/(node_label_freq[i][edge[0]]*(node_label_freq[i][edge[1]]-1))
                            ej = 2*edges[j][edge]/(node_label_freq[j][edge[0]]*(node_label_freq[j][edge[1]]-1))
                        else:
                            ei = 2*edges[i][edge]/(node_label_freq[i][edge[0]]*node_label_freq[i][edge[1]])
                            ej = 2*edges[j][edge]/(node_label_freq[j][edge[0]]*node_label_freq[j][edge[1]])
                        m1 = min(node_label_freq[i][edge[0]], node_label_freq[j][edge[0]])
                        m2 = min(node_label_freq[i][edge[1]], node_label_freq[j][edge[1]])
                        K[i,j] += sqrt(ei)*sqrt(ej)*m1*m2
                K[j,i] = K[i,j]

    return K


if args.dataset in ["MUTAG", "DD", "NCI1", "PROTEINS", "ENZYMES"]:
    use_node_labels = True
else:
    use_node_labels = False

print('Loading dataset')
Gs, y = load_data(args.dataset, use_node_labels)

print('Computing kernel')
node_labels, max_label_counts = get_wl_labels(Gs, args.T, use_node_labels)
K = compute_gawl_kernel(Gs, args.T, node_labels, max_label_counts)

print('Evaluating kernel')
eval_10_fold(args.dataset, K, y)