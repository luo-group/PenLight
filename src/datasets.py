# -*- codeing = utf-8 -*-
# @Time:  11:48 上午
# @Author: Jiaqi Luo
# @File: datasets.py
# @Software: PyCharm
import torch
import torch_geometric
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch_cluster
import numpy as np

from pathlib import Path
import math
import time
import random
import copy
import h5py, json
import tqdm
from utils import seq2blosum50, seq2onehot

# data_root = "../data/" # Changed for EC
# data_root = "../ec_data/"

def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class TripleDataGVP(torch_geometric.data.Data):
    def __init__(self, x_anchor, seq_anchor, name_anchor, node_s_anchor, node_v_anchor, edge_s_anchor,
                 edge_v_anchor, edge_index_anchor, mask_anchor, x_pos, seq_pos, name_pos, node_s_pos,
                 node_v_pos, edge_s_pos, edge_v_pos, edge_index_pos, mask_pos, x_neg, seq_neg, name_neg,
                 node_s_neg, node_v_neg, edge_s_neg, edge_v_neg, edge_index_neg, mask_neg, label, pos_sim):
        super().__init__()
        self.x_anchor, self.x_pos, self.x_neg = x_anchor, x_pos, x_neg
        self.seq_anchor, self.seq_pos, self.seq_neg = seq_anchor, seq_pos, seq_neg
        self.name_anchor, self.name_pos, self.name_neg = name_anchor, name_pos, name_neg
        self.node_s_anchor, self.node_s_pos, self.node_s_neg = node_s_anchor, node_s_pos, node_s_neg
        self.node_v_anchor, self.node_v_pos, self.node_v_neg = node_v_anchor, node_v_pos, node_v_neg
        self.edge_s_anchor, self.edge_s_pos, self.edge_s_neg = edge_s_anchor, edge_s_pos, edge_s_neg
        self.edge_v_anchor, self.edge_v_pos, self.edge_v_neg = edge_v_anchor, edge_v_pos, edge_v_neg
        self.edge_index_anchor, self.edge_index_pos, self.edge_index_neg = edge_index_anchor, edge_index_pos, \
                                                                           edge_index_neg
        self.mask_anchor, self.mask_pos, self.mask_neg = mask_anchor, mask_pos, mask_neg
        self.label = label
        self.pos_sim = pos_sim

    def __inc__(self, key, value, *args, **kwargs):
        features = ['seq', 'name', 'node_s', 'node_v', 'edge_s', 'edge_v', 'edge_index']
        anchor_features = [f + '_anchor' for f in features]
        pos_features = [f + '_pos' for f in features]
        neg_features = [f + '_neg' for f in features]
        if key in anchor_features:
            return self.x_anchor.size(0)
        elif key in pos_features:
            return self.x_pos.size(0)
        elif key in neg_features:
            return self.x_neg.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class TripleDataGNN(torch_geometric.data.Data):
    def __init__(self, x_anchor, edge_index_anchor, edge_attr_anchor, seq_anchor, name_anchor, x_pos,
                 edge_index_pos, edge_attr_pos, seq_pos, name_pos, x_neg, edge_index_neg, edge_attr_neg,
                 seq_neg, name_neg, label, pos_sim):
        super().__init__()
        self.x_anchor, self.x_pos, self.x_neg = x_anchor, x_pos, x_neg
        self.edge_index_anchor, self.edge_index_pos, self.edge_index_neg = edge_index_anchor, edge_index_pos, \
                                                                           edge_index_neg
        self.edge_attr_anchor, self.edge_attr_pos, self.edge_attr_neg = edge_attr_anchor, edge_attr_pos, edge_attr_neg
        self.name_anchor, self.name_pos, self.name_neg = name_anchor, name_pos, name_neg
        self.seq_anchor, self.seq_pos, self.seq_neg = seq_anchor, seq_pos, seq_neg
        self.label = label
        self.pos_sim = pos_sim

    def __inc__(self, key, value, *args, **kwargs):
        features = ['edge_index', 'edge_attr', 'name', 'seq']
        anchor_features = [f + '_anchor' for f in features]
        pos_features = [f + '_pos' for f in features]
        neg_features = [f + '_neg' for f in features]
        if key in anchor_features:
            return self.x_anchor.size(0)
        elif key in pos_features:
            return self.x_pos.size(0)
        elif key in neg_features:
            return self.x_neg.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ProteinGraphDataset(Dataset):
    def __init__(self, train, datasplitter, n_classes=4, num_positional_embeddings=16,
                 top_k=30, num_rbf=16, balanced_sampling=False, graph_type='gat', seq_emb="esm", 
                 device="cpu", edge_type="topk", edge_threshold=8.0, scalar_only=False):
        super(ProteinGraphDataset, self).__init__()
        self.data_dict = train
        # self.data_list = list(self.data_dict.values())
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.graph_type = graph_type
        self.edge_type = edge_type
        self.edge_threshold = edge_threshold
        self.seq_emb = seq_emb
        self.device = device
        self.scalar_only = scalar_only
        self.node_counts = [len(self.data_dict[k]['seq']) for k in self.data_dict.keys()]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12, 'X': 20, 'U': 21, 'B': 22,
                              'Z': 23, 'J': 24, 'O': 25}  # Modified: added 'X': 0
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        for i in list(self.data_dict.keys()):
            coords = self.data_dict[i]["coordinates"]
            self.data_dict[i]['coordinates'] = list(zip(coords['N'], coords['CA'], coords['C'], coords['O']))

        self.balanced_sampling = balanced_sampling
        self.seq_id, _ = zip(*[(seq_id, info) for seq_id, info in train.items()])
        self.id2label, self.label2id = datasplitter.parse_label_mapping_cath(set(train.keys()))

        # if classes should be sampled evenly (not all training samples are used in every epoch)
        if self.balanced_sampling:
            print("Using balanced sampling!")
            self.unique_labels = self.get_unique_labels()
            self.data_len = len(self.unique_labels)
        else:  # if you want to iterate over all training samples
            self.data_len = len(self.seq_id)

        self.id2graphs = self.get_graphs(list(train.keys()))
        self.n_classes = n_classes  # number of class levels

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.balanced_sampling:  # get a CATH class, instead of a trainings sample
            c, a, t, h = self.unique_labels[index]  # get CATH class
            anchor_candidates = self.label2id[c][a][t][h]  # get samples within this CATH class
            anchor_id = random.choice(anchor_candidates)  # randomly pick one of these samples as anchor
            anchor = self.id2graphs[anchor_id]  # retrieve graph for this sample
            anchor_label = self.id2label[anchor_id]  # retrieve label for this sample
        else:  # get a training sample (over-samples large CATH families according to occurance)
            anchor = self.id2graphs[list(self.id2graphs.keys())[index]]  # get graph of anchor
            anchor_id = self.seq_id[index]  # get CATH ID of anchor
            anchor_label = self.id2label[anchor_id]  # get CATH label of anchor
        pos, neg, pos_label, neg_label, pos_sim = self.get_pair(anchor_id, anchor_label)
        t_anc = torch.tensor(anchor_label).view(1, -1)
        t_pos = torch.tensor(pos_label).view(1, -1)
        t_neg = torch.tensor(neg_label).view(1, -1)
        label = torch.cat([t_anc, t_pos, t_neg], dim=0).view(1, 3, -1)
        triplet = None
        if self.graph_type == 'gvp':
            triplet = TripleDataGVP(anchor.x, anchor.seq, anchor.name, anchor.node_s, anchor.node_v, anchor.edge_s,
                                    anchor.edge_v, anchor.edge_index, anchor.mask, pos.x, pos.seq, pos.name,
                                    pos.node_s, pos.node_v, pos.edge_s, pos.edge_v, pos.edge_index, pos.mask,
                                    neg.x, neg.seq, neg.name, neg.node_s, neg.node_v, neg.edge_s, neg.edge_v,
                                    neg.edge_index, neg.mask, label, pos_sim)
        else:
            triplet = TripleDataGNN(anchor.x, anchor.edge_index, anchor.edge_attr, anchor.seq,
                                    anchor.name, pos.x, pos.edge_index, pos.edge_attr, pos.seq,
                                    pos.name, neg.x, neg.edge_index, neg.edge_attr, neg.seq, neg.name,
                                    label, pos_sim)
            triplet.num_nodes = len(anchor.x) + len(pos.x) + len(neg.x)
        return triplet

    def get_id2graphs(self):
        return self.id2graphs

    def get_label_vector(self, anchor_label, pos_label, neg_label):
        anc = torch.tensor(anchor_label).view(1, -1)
        pos = torch.tensor(pos_label).view(1, -1)
        neg = torch.tensor(neg_label).view(1, -1)
        y = torch.cat([anc, pos, neg], dim=0)
        return y.view(1, 3, -1)

    def get_unique_labels(self):
        unique_set = set()
        unique_labels = list()
        for _, cath_label in self.id2label.items():
            cath_str = '.'.join([str(cath_hierarchy_lvl)
                                 for cath_hierarchy_lvl in cath_label])
            if cath_str in unique_set:
                continue
            unique_labels.append(cath_label)
            unique_set.add(cath_str)
        print("Number of unique CATH labels in train: {}".format(len(unique_set))) # Changed for EC
        # print("Number of unique EC labels in train: {}".format(len(unique_set)))
        return unique_labels

    def get_rnd_label(self, labels, is_pos, existing_label=None):
        n_labels = len(labels)
        # if alternative labels are available, ensure difference between existing and new label
        if n_labels > 1 and existing_label is not None:
            labels = [label for label in labels if label != existing_label]
            n_labels -= 1

        rnd_idx = np.random.randint(0, n_labels)

        i = iter(labels)
        for _ in range(rnd_idx):
            next(i)
        rnd_label = next(i)
        # do not accidentaly draw the same label; instead draw again if necessary
        if existing_label is not None and rnd_label == existing_label:
            if is_pos:  # return the label itself for positives
                # Allow positives to have the same class as the anchor (relevant for rare classes)
                return existing_label
            else:
                # if there exists no negative sample for a certain combination of anchor and similarity-level
                return None
        return rnd_label

    def get_rnd_candidates(self, anchor_label, similarity_level, is_pos):

        # Get CATH classification of anchor sample
        class_n, arch, topo, homo = anchor_label

        if similarity_level == 0:  # No similarity - different class
            rnd_class = self.get_rnd_label(
                self.label2id.keys(), is_pos, class_n)
            rnd_arch = self.get_rnd_label(
                self.label2id[rnd_class].keys(), is_pos)
            rnd_topo = self.get_rnd_label(
                self.label2id[rnd_class][rnd_arch].keys(), is_pos)
            rnd_homo = self.get_rnd_label(
                self.label2id[rnd_class][rnd_arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[rnd_class][rnd_arch][rnd_topo][rnd_homo]

        elif similarity_level == 1:  # Same class but different architecture
            rnd_arch = self.get_rnd_label(
                self.label2id[class_n].keys(), is_pos, arch)
            rnd_topo = self.get_rnd_label(
                self.label2id[class_n][rnd_arch].keys(), is_pos)
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][rnd_arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[class_n][rnd_arch][rnd_topo][rnd_homo]

        elif similarity_level == 2:  # Same Class & Architecture but different topo
            rnd_topo = self.get_rnd_label(
                self.label2id[class_n][arch].keys(), is_pos, topo)
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[class_n][arch][rnd_topo][rnd_homo]

        elif similarity_level == 3:  # Same Class & Architecture & topo but different homo
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][arch][topo].keys(), is_pos, homo)
            candidates = self.label2id[class_n][arch][topo][rnd_homo]

        # Highest similarity - different homology class (only relevent for positives)
        elif similarity_level == 4:
            candidates = self.label2id[class_n][arch][topo][homo]

        else:
            raise NotImplementedError

        return candidates

    def check_triplet(self, anchor_label, pos_label, neg_label, neg_hardness, pos_hardness):
        assert neg_hardness < pos_hardness, print("Neg sample more similar than pos sample")

        for i in range(0, pos_hardness):
            assert anchor_label[i] == pos_label[i], print("Pos label not overlapping:\n" +
                                                          "Diff: {}\nanchor:{}\npos:{}\nneg:{}".format(pos_hardness,
                                                                                                       anchor_label,
                                                                                                       pos_label,
                                                                                                       neg_label))
        for j in range(0, neg_hardness):
            assert anchor_label[j] == neg_label[j], print("Neg label not overlapping:\n" +
                                                          "Diff: {}\nanchor:{}\npos:{}\nneg:{}".format(neg_hardness,
                                                                                                       anchor_label,
                                                                                                       pos_label,
                                                                                                       neg_label))
        assert anchor_label[neg_hardness] != neg_label[neg_hardness], print(
            "Neg label not different from anchor")
        return None

    def get_pair(self, anchor_id, anchor_label, hardness_level=None, verbose=False):
        pos, neg = None, None
        pos_label, neg_label = None, None

        while pos is None or neg is None:
            neg_similarity = np.random.randint(self.n_classes)
            pos_similarity = neg_similarity + 1
            try:
                neg_candidates = self.get_rnd_candidates(
                    anchor_label, neg_similarity, is_pos=False)  # get set of negative candidates
                neg_id = random.choice(neg_candidates)  # randomly pick one of the neg. candidates
                neg_label = self.id2label[neg_id]  # get label of randomly picked neg.
                neg = self.id2graphs[neg_id]  # get embedding of randomly picked neg.

                # repeat the same for the positive sample
                pos_candidates = self.get_rnd_candidates(
                    anchor_label, pos_similarity, is_pos=True)
                pos_id = random.choice(pos_candidates)

                # ensure that we do not randomly pick the same protein as anchor and positive
                if pos_id == anchor_id and len(pos_candidates) > 1:
                    while pos_id == anchor_id:  # re-draw from the pos. candidates if possible
                        pos_id = random.choice(pos_candidates)
                # if there is only one protein in a superfamily (anchor==positive without other candidates),
                # re-start picking process
                elif pos_id == anchor_id and len(pos_candidates) == 1:
                    continue

                pos = self.id2graphs[pos_id]
                pos_label = self.id2label[pos_id]
                # if we successfully picked anchor, positive and negative candidates, do same sanity checks
                if pos_label is not None and neg_label is not None:
                    self.check_triplet(anchor_label, pos_label,
                                       neg_label, neg_similarity, pos_similarity)
                else:  # if no triplet could be formed for a given combination of similarities/classes
                    continue

            except NotImplementedError:  # if you try to create triplets for a class level that is not yet
                # implemented in get_rnd_candidates
                print(anchor_id, anchor_label)
                raise NotImplementedError

            except KeyError:
                # if get_rnd_label returned None because no negative could be found
                # for a certain combination of anchor protein and similarity-lvl
                # re-start picking process
                continue

        if verbose:
            print('#### Example ####')
            print('Anc ({}) label: {}'.format(anchor_id, anchor_label))
            print('Pos ({}) label: {}'.format(pos_id, self.id2label[pos_id]))
            print('Neg ({}) label: {}'.format(neg_id, self.id2label[neg_id]))
            print('#### Example ####')

        return pos, neg, pos_label, neg_label, pos_similarity

    def get_example(self):
        example_id = next(iter(self.id2graphs.keys()))
        example_label = self.id2label[example_id]
        self.get_pair(example_id, example_label, verbose=True)
        return None

    def get_single_example(self, cath_id):
        return self.featurize_as_graph(self.data_dict[cath_id], cath_id, self.graph_type)

    def get_graphs(self, cath_ids):
        id2graph = dict()
        for i in tqdm.tqdm(cath_ids):
        # for i in cath_ids:
            id2graph[i] = self.featurize_as_graph(self.data_dict[i], i, self.graph_type)

        return id2graph

    def featurize_as_graph(self, protein, cath_id, graph_type='gat'):
        name = cath_id
        # print(name)
        with torch.no_grad():
            coords = torch.as_tensor(protein['coordinates'], dtype=torch.float32)
            # print(coords.size())
            # seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], dtype=torch.long)
            seq = protein['seq']

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            if self.edge_type == "topk":
                edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            elif self.edge_type == '8A':
                edge_index = torch_cluster.radius_graph(X_ca, r=self.edge_threshold)
            else:
                raise NotImplementedError

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))
        data = None
        if graph_type == 'gvp':
            data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                             node_s=node_s, node_v=node_v,
                                             edge_s=edge_s, edge_v=edge_v,
                                             edge_index=edge_index, mask=mask)
        else:
            if self.scalar_only:
                x = node_s
                edge_attr = edge_s
            else:
                x = torch.cat([X_ca, node_s, node_v.reshape((node_v.shape[0], -1))], dim=1)
                edge_attr = torch.cat([edge_s, edge_v.reshape((edge_v.shape[0], -1))], dim=1)
            if self.seq_emb == "onehot":
                embedding = seq2onehot(seq)
                x = torch.cat([x, embedding], dim=1)
            if self.seq_emb == "blosum50":
                embedding = seq2blosum50(seq)
                x = torch.cat([x, embedding], dim=1)
            if self.seq_emb == "both" or self.seq_emb == "all":
                embedding = seq2onehot(seq)
                x = torch.cat([x, embedding], dim=1)
                embedding = seq2blosum50(seq)
                x = torch.cat([x, embedding], dim=1)
            # x = x / torch.linalg.norm(x)
            # edge_attr = edge_attr / torch.linalg.norm(edge_attr)
            data = torch_geometric.data.Data(x=x, edge_index=edge_index, seq=seq, name=name, edge_attr=edge_attr)
        return data

    def _check(self, X_ca, edge_index):
        num_nodes = len(X_ca)
        max_dist = torch.zeros(num_nodes)
        num_edges = len(edge_index[0])
        for i in range(num_edges):
            max_dist[edge_index[0][i]] = max(torch.dist(X_ca[edge_index[0][i]], X_ca[edge_index[1][i]]), max_dist[edge_index[0][i]])
            max_dist[edge_index[1][i]] = max(torch.dist(X_ca[edge_index[0][i]], X_ca[edge_index[1][i]]), max_dist[edge_index[1][i]])
        # print(max_dist.min().item())
        return max_dist.min().item() >= 8

    def _get_edges_threshold(self, X_ca, threshold=8.0):
        edge_index = [[], []]
        knn_edges = torch_cluster.knn_graph(X_ca, k=self.top_k)
        num_knn_edges = len(knn_edges[0])
        for i in num_knn_edges:
            if torch.dist(X_ca[knn_edges[0][i]], X_ca[knn_edges[1][i]]) <= threshold:
                edge_index[0].append(knn_edges[0][i])
                edge_index[1].append(knn_edges[1][i])
        
        return torch.tensor(edge_index)

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


class DataSplitter:
    def __init__(self, json_file, data_root, task='cath', verbose=True):
        self.verbose = verbose
        self.data_dir = data_root
        self.id2info = self.get_preprocessed_info(json_file)
        self.task = task
        if verbose:
            print('Loaded info for n_proteins: {}'.format(
                len(self.id2info)))
        if task == 'cath':
            self.cath_label_path = self.data_dir / 'cath-domain-list-S100.txt' # Changed for EC
        else:
            self.cath_label_path = self.data_dir / 'pdb_ec_chain.txt'
        self.id2label, self.label2id = self.parse_label_mapping_cath(
            set(self.id2info.keys()))

    def get_id2info(self):
        return self.id2info

    def parse_label_mapping_cath(self, id_subset):
        id2label = dict()
        label2id = dict()
        with open(self.cath_label_path, 'r') as f:
            for n_domains, line in enumerate(f):

                # skip header lines
                if line.startswith("#"):
                    continue

                data = line.split()
                identifier = data[0]
                # skip annotations of proteins without embedding (not part of data set)
                if identifier not in id_subset:
                    continue

                cath_class = int(data[1])
                cath_arch = int(data[2])
                cath_topo = int(data[3])
                cath_homo = int(data[4])

                if cath_class not in label2id:
                    label2id[cath_class] = dict()
                if cath_arch not in label2id[cath_class]:
                    label2id[cath_class][cath_arch] = dict()
                if cath_topo not in label2id[cath_class][cath_arch]:
                    label2id[cath_class][cath_arch][cath_topo] = dict()
                if cath_homo not in label2id[cath_class][cath_arch][cath_topo]:
                    label2id[cath_class][cath_arch][cath_topo][cath_homo] = list()

                id2label[identifier] = [cath_class,
                                        cath_arch, cath_topo, cath_homo]
                label2id[cath_class][cath_arch][cath_topo][cath_homo].append(
                    identifier)

        if self.verbose:
            print('Finished parsing n_domains: {}'.format(n_domains))
            print("Total length of id2label: {}".format(len(id2label)))
        return id2label, label2id

    def read_cath_ids(self, path):
        ids = set()
        id_list = list()
        seq_test = dict()

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    line = line.replace(">", "")
                    if '|' in line:
                        seq_id = line.split('|')[2]
                    else:
                        seq_id = line
                    if seq_id in ids:  # some weird double entries in CATH test set..
                        continue
                    ids.add(seq_id)
                    id_list.append(seq_id)
                    seq_test[seq_id] = list()
                else:
                    seq_test[seq_id].append(line)

        # some identical sequences need to be removed
        seq_set = {''.join(seq): seq_id for seq_id, seq in seq_test.items()}
        id_list = [seq_id for seq, seq_id in seq_set.items()]

        # assert that no identical seqs are in the sets
        assert len(seq_set) == len(id_list)
        if self.verbose:
            print('Example CATH ID: {}'.format(seq_id)) # Changed for EC
            # print('Example EC ID: {}'.format(seq_id))
            print('-- Loaded {} proteins from {}'.format(len(id_list), path))
        return id_list

    def get_preprocessed_info(self, json_file):
        dataset = {}
        with open(json_file, "r") as f:
            dataset = json.loads(f.read())

        return dataset

    def get_infos(self, json_file):
        infos = dict()
        with open(json_file, 'r') as f:
            infos = json.loads(f.read())

        return infos

    # TODO: modify this function, we just use nrS40 as our source
    def get_predef_splits(self, p_train=None, p_test=None):

        if p_train is None or p_test is None:
            if self.task == 'cath':
                p_train = self.data_dir / "splits_json/train.json" # Changed for EC
                p_val = self.data_dir / "splits_json/val.json" # Changed for EC
                p_valLookup20 = self.data_dir / "splits_json/train.json" # Changed for EC
            elif self.task == 'ec':
                p_train = self.data_dir / "splits_json/train.json"
                p_val = self.data_dir / "splits_json/valid.json"
                p_valLookup20 = self.data_dir / "splits_json/train.json"
            else:
                raise NotImplementedError

        val = self.get_infos(p_val)
        valLookup20 = self.get_infos(p_valLookup20)
        train = self.get_infos(p_train)

        if self.verbose:
            print('##########')
            print('Finished splitting data!')
            print('Train set size: {}'.format(len(train)))
            print('Val set size: {}'.format(len(val)))
            print('ValLookup20 size: {}'.format(len(valLookup20)))
            print('##########')
        return train, val, valLookup20

    def get_test_splits(self, test_file=None):
        if self.task == 'cath':
            p_test = self.data_dir / "splits_json/test.json" # Changed for EC
            p_lookup = self.data_dir / "splits_json/lookup.json" # Changed for EC
        elif self.task == 'ec':
            if not test_file:
                p_test = self.data_dir / "splits_json/test95.json"
            else:
                p_test = test_file
            p_lookup = self.data_dir / "splits_json/train.json"

        test = self.get_infos(p_test)
        lookup = self.get_infos(p_lookup)
        if self.verbose:
            print('##########')
            print('Finished splitting data!')
            print('Test set size: {}'.format(len(test)))
            print('Lookup set size: {}'.format(len(lookup)))
            print('##########')

        return test, lookup
