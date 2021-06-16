import json
import math

import torch
import pickle
import torch.utils.data
import time
import dgl
import numpy as np
import csv
from scipy import sparse as sp


class JCCHTPCHDGL(torch.utils.data.Dataset):
    def __init__(self,  name, split):
        if name == "compact-jcch":
            self.train_path = "data/JCCHTPCH/train_q_jcch_db_jcch/compact/"
            self.test_path = "data/JCCHTPCH/test_q_jcch_db_jcch/compact/"
            self.train_col = 2
            self.test_col = 0
        self.split = split
        self.train_cardinalities_path = "data/JCCHTPCH/train_cardinalities.csv"
        self.test_cardinalities_path = "data/JCCHTPCH/cardinalities_test_sets.csv"
        self.graph_labels = []
        self.graph_lists = []
        self._prepare()

    def _prepare(self):
        print("preparing graphs for the %s set..." % (self.split.upper()))
        with open("data/JCCHTPCH/shuffled_index_1_to_5000.json") as json_file:
            shuffled_index = json.load(json_file)["shuffled index 1 to 5000"]
        with open(self.test_cardinalities_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            cardinalities_test = []
            for row in reader:
                card = int(row[self.test_col])
                if card > 0:
                    card = math.log(card)
                cardinalities_test.append(card)
        with open(self.train_cardinalities_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            cardinalities_train = []
            for row in reader:
                card = int(row[self.test_col])
                if card > 0:
                    card = math.log(card)
                cardinalities_train.append(card)
        min_el = min(min(cardinalities_train), min(cardinalities_test))
        max_el = max(max(cardinalities_train), max(cardinalities_test))
        for i in range(len(cardinalities_train)):
            cardinalities_train[i] = (cardinalities_train[i] - min_el) / (max_el - min_el)
        for i in range(len(cardinalities_test)):
            cardinalities_test[i] = (cardinalities_test[i] - min_el) / (max_el - min_el)
        if self.split == "train":
            name = self.train_path
            self.n_samples = 4250
            r = range(0, 4250)
            cardinalities = cardinalities_train
        elif self.split == "val":
            name = self.train_path
            self.n_samples = 750
            r = range(4250, 5000)
            cardinalities = cardinalities_train
        else:
            name = self.test_path
            self.n_samples = 22
            r = range(1, 23)
            cardinalities = cardinalities_test
        for i in r:
            if self.split != "test":
                shuffled_i = shuffled_index[i]
            else:
                shuffled_i = i
            s = name + str(shuffled_i) + '.pkl'
            with open(s, "rb") as f:
                g = pickle.load(f)
            g2 = dgl.graph(g.edges())
            g2.ndata["feat"] = g.ndata["feat"].float()
            g2.edata["feat"] = g.edata["feat"].float()

            self.graph_lists.append(g2)
            self.graph_labels.append(torch.tensor(cardinalities[shuffled_i - 1]))



    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class JCCHTPCHDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name):
        t0 = time.time()
        self.train = JCCHTPCHDGL(name, 'train')
        self.test = JCCHTPCHDGL(name, 'test')
        self.val = JCCHTPCHDGL(name, 'val')
        print("Time taken: {:.4f}s".format(time.time() - t0))



class JCCHTPCHDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        path_to_datasets = "data/JCCHTPCH/"
        with open(path_to_datasets + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = 0
            self.num_bond_type = 0
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]




def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()
    return g
