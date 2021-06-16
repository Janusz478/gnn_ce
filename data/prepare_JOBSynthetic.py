import numpy as np
import torch
import pickle
import time
import os
import matplotlib.pyplot as plt

import pickle
from JOBSynthetic import JOBSyntheticDatasetDGL
from JOBSynthetic import JOBSyntheticDataset
from data import LoadData
from torch.utils.data import DataLoader
#os.chdir('..')
print(os.getcwd())

#DATASET_NAME = "Logical"
DATASET_NAME = "Logical-Samples"
#DATASET_NAME = "Compact"
#DATASET_NAME = "Compact-Samples"
#DATASET_NAME = "Fine-Grained"
#DATASET_NAME = "Physical"
#DATASET_NAME = "Query-Plan-Oriented"

dataset = JOBSyntheticDatasetDGL(DATASET_NAME)

def plot_histo_graphs(dataset, title):
    # histogram of graph sizes
    graph_sizes = []
    for graph in dataset:
        graph_sizes.append(graph[0].number_of_nodes())
    plt.figure(1)
    plt.hist(graph_sizes, bins=20)
    plt.title(title)
    plt.show()
    graph_sizes = torch.Tensor(graph_sizes)
    print('min/max :', graph_sizes.min().long().item(), graph_sizes.max().long().item())


#plot_histo_graphs(dataset.train, 'trainset')
#plot_histo_graphs(dataset.val, 'valset')
#plot_histo_graphs(dataset.test, 'testset')


print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])

num_atom_type = 0   #irrelevant
num_bond_type = 0   #irrelevant

start = time.time()
with open('JOBSynthetic/' + DATASET_NAME + ".pkl", 'wb') as f:
        pickle.dump([dataset.train,dataset.val,dataset.test,num_atom_type,num_bond_type],f)
print('Time (sec):',time.time() - start)

dataset = LoadData(DATASET_NAME)
print(dataset)
trainset, valset, testset = dataset.train, dataset.val, dataset.test
print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])

batch_size = 10
collate = JOBSyntheticDataset.collate
print(JOBSyntheticDataset)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)
#dataset._add_self_loops()
#dataset._add_positional_encodings(8)