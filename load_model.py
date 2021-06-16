import csv
import json
import math
import os

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from data.data import LoadData
from torch.utils.data import DataLoader
from nets.molecules_graph_regression.load_net import gnn_model


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return qerror

def print_errors(preds_unnorm, labels_unnorm):
    preds_unnorm = torch.FloatTensor(preds_unnorm)
    labels_unnorm = torch.FloatTensor(labels_unnorm)
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    print()
    print("The Mean Absolute error is {}".format(mae_loss(preds_unnorm, labels_unnorm)))
    print("The Mean Squared error is {}".format(mse_loss(preds_unnorm, labels_unnorm)))
    return print_qerror(preds_unnorm, labels_unnorm)


def get_net_params(device, dataset):
    seed = 41; epochs = 1000; batch_size = 32; init_lr = 0.003; lr_reduce_factor = 0.5; lr_schedule_patience = 10; min_lr = 1e-5; weight_decay = 0
    L = 16; hidden_dim = 300; out_dim = hidden_dim; dropout = 0.0; readout = 'mean'
    n_heads = -1
    edge_feat = True;
    # edge_feat = False
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    # self_loop = True
    max_time = 12
    pos_enc = False
    # pos_enc = True
    pos_enc_dim = 2

    net_params = {}
    net_params['device'] = device
    net_params['num_atom_type'] = 0
    net_params['num_bond_type'] = 0
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "sum"
    net_params['layer_norm'] = False
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop

    # added
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)

    # for MLPNet
    net_params['gated'] = gated

    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
    net_params['kernel'] = kernel

    # specific for GIN
    net_params['n_mlp_GIN'] = n_mlp_GIN
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'

    # specific for graphsage
    net_params['sage_aggregator'] = 'meanpool'

    # specific for diffpoolnet
    net_params['data_mode'] = 'default'
    net_params['gnn_per_block'] = gnn_per_block
    net_params['embedding_dim'] = embedding_dim
    net_params['pool_ratio'] = pool_ratio
    net_params['linkpred'] = True
    net_params['num_pool'] = 1
    net_params['cat'] = False
    net_params['batch_size'] = batch_size

    # specific for pos_enc_dim
    net_params['pos_enc'] = pos_enc
    net_params['pos_enc_dim'] = pos_enc_dim
    return net_params


def get_table_error_barchart(errors, DATASET_NAME):
    labels = []
    with open("out/mscn_deepdb_results/cardinalities_mscn.csv") as f_mscn:
        reader = csv.reader(f_mscn, delimiter=',')
        next(reader, None)
        mscn_error = []
        for row in reader:
            mscn_error.append(float(row[0]))
            labels.append(float(row[1]))
    print("MSCN Job Light Error is:")
    mscn_error = print_errors(mscn_error, labels)

    with open("out/mscn_deepdb_results/imdb_light_model_based_budget_5.csv") as f_deepdb:
        deepdb_error = []
        reader = csv.reader(f_deepdb, delimiter=',')
        next(reader, None)
        for row in reader:
            deepdb_error.append(float(row[2]))

    print("DeepDB Job Light Error is:")
    deepdb_error = print_errors(deepdb_error, labels)
    with open("out/mscn_deepdb_results/query_numbers_tables_predicates.json") as j_file:
        query_numbers = json.load(j_file)
    mscn_table_mean = []
    deep_db_table_mean = []
    gnn_table_mean = []
    que_list = []
    mscn_list = []
    deep_db_list = []
    gnn_list = []
    for table in query_numbers.values():
        for column in table.values():
            for number in column:
                que_list.append(number)
        for query_number in que_list:
            mscn_list.append(mscn_error[query_number-1])
            deep_db_list.append(deepdb_error[query_number-1])
            gnn_list.append(errors[query_number-1])
        mscn_table_mean.append(np.median(mscn_list))
        deep_db_table_mean.append(np.median(deep_db_list))
        gnn_table_mean.append(np.median(gnn_list))
        mscn_list.clear()
        deep_db_list.clear()
        gnn_list.clear()
        que_list.clear()

    table_labels = ["2", "3", "4", "5"]
    x = np.arange(len(table_labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, mscn_table_mean, width, label='MSCN')
    ax.bar(x + width, deep_db_table_mean, width, label='DeepDB')
    ax.bar(x + 2 * width, gnn_table_mean, width, label='GNN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Median Q-Error')
    ax.set_xlabel('Tables')
    ax.set_title('Cardinality Estimation Errors per Join Size')
    ax.set_xticks(x)
    ax.set_xticklabels(table_labels)
    #ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()


    plt.savefig("out/Figures/{}-tables.png".format(DATASET_NAME))
    plt.show()


def get_table_nr_preds_error_barchart(errors, DATASET_NAME):
    labels = []
    with open("out/mscn_deepdb_results/cardinalities_mscn.csv") as f_mscn:
        reader = csv.reader(f_mscn, delimiter=',')
        next(reader, None)
        mscn_error = []
        for row in reader:
            mscn_error.append(float(row[0]))
            labels.append(float(row[1]))
    print("MSCN Job Light Error is:")
    mscn_error = print_errors(mscn_error, labels)

    with open("out/mscn_deepdb_results/imdb_light_model_based_budget_5.csv") as f_deepdb:
        deepdb_error = []
        reader = csv.reader(f_deepdb, delimiter=',')
        next(reader, None)
        for row in reader:
            deepdb_error.append(float(row[2]))

    print("DeepDB Job Light Error is:")
    deepdb_error = print_errors(deepdb_error, labels)
    with open("out/mscn_deepdb_results/query_numbers_tables_predicates.json") as j_file:
        query_numbers = json.load(j_file)
    mscn_table_mean = []
    deep_db_table_mean = []
    gnn_table_mean = []
    que_list = []
    mscn_list = []
    deep_db_list = []
    gnn_list = []
    for table in query_numbers.values():
        for column in table.values():
            if len(column) > 0:
                for number in column:
                    que_list.append(number)
                for query_number in que_list:
                    mscn_list.append(mscn_error[query_number - 1])
                    deep_db_list.append(deepdb_error[query_number - 1])
                    gnn_list.append(errors[query_number - 1])
                mscn_table_mean.append(np.median(mscn_list))
                deep_db_table_mean.append(np.median(deep_db_list))
                gnn_table_mean.append(np.median(gnn_list))
                mscn_list.clear()
                deep_db_list.clear()
                gnn_list.clear()
                que_list.clear()

    table_labels = ["2-2", "3-1", "3-2", "3-3", "3-4", "4-1", "4-2", "4-3", "4-4", "4-5", "5-2", "5-3", "5-4", "5-5"]
    x = np.arange(len(table_labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, mscn_table_mean, width, label='MSCN')
    ax.bar(x + width, deep_db_table_mean, width, label='DeepDB')
    ax.bar(x + 2 * width, gnn_table_mean, width, label='GNN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Median Q-Error')
    ax.set_xlabel('Tables')
    ax.set_title('Cardinality Estimation Errors per Join Size')
    ax.set_xticks(x)
    ax.set_xticklabels(table_labels)
    # ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()

    plt.savefig("out/Figures/{}-tables-columns.png".format(DATASET_NAME))
    plt.show()



def printPredictions(model, device, data_loader, labels, card_start, card_end):
    #print()
    #print("Printing cardinalities of {} from {} to {}".format(labels, card_start, card_end))
    if labels == "job-light":
        csvpath = 'data/job-light-pickle/team10_job_light_cardinalities.csv'
    elif labels == "synthetic":
        csvpath = 'data/synthetic-pickle/team10_synthetic_cardinalities.csv'
    elif labels == "synthetic-big":
        csvpath = 'data/synthetic-big-pickle/team10_synthetic_big_cardinalities.csv'

    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        label_unnormalized = []
        for row in reader:
            label_unnormalized.append(float(row[1]))

    min_value = 0.0
    max_value = 22.97847740248224
    label_unnormalized = label_unnormalized[card_start:card_end]
    predictions_unnormalized = []
    model.eval()
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            predictions_normalized = batch_scores.data.cpu().numpy()
            for el in predictions_normalized:
                predictions_unnormalized.append(math.exp(el * (max_value - min_value) + min_value))

    return predictions_unnormalized, label_unnormalized


def load_and_predict(model_path, DATASET_NAME, batch_size, hidden_dim):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if torch.cuda.is_available():
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")

    if DATASET_NAME == "Logical" or DATASET_NAME == "Logical-Big" or DATASET_NAME == "Compact-Big" or DATASET_NAME == "Compact" or DATASET_NAME == "Logical-Samples" or DATASET_NAME == "Logical-Big-Samples":
        if DATASET_NAME == "Logical" or DATASET_NAME == "Logical-Big":
            big = "Logical-Big"
            small = "Logical"
        elif DATASET_NAME == "Compact" or DATASET_NAME == "Compact-Big":
            big = "Compact-Big"
            small = "Compact"
        elif DATASET_NAME == "Logical-Samples" or DATASET_NAME == "Logical-Big-Samples":
            big = "Logical-Big-Samples"
            small = "Logical-Samples"


        """dataset = LoadData(big)
        trainset, valset, testset = dataset.train, dataset.val, dataset.test
        model = gnn_model("GatedGCN", get_net_params(device, dataset))
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False,
                                  collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False,
                                collate_fn=dataset.collate)
        train_pred, train_labels = printPredictions(model, device, train_loader, "synthetic-big", 0, 90000)
        val_pred, val_labels = printPredictions(model, device, val_loader, "synthetic-big", 90000, 100000)
        train_pred.extend(val_pred)
        train_labels.extend(val_labels)
        print("\nThe Errors of the Big Synthetic is: ")
        print_errors(train_pred, train_labels)"""


        dataset = LoadData(small)
        trainset, valset, testset = dataset.train, dataset.val, dataset.test
        net_param = get_net_params(device, dataset)
        net_param['hidden_dim'] = hidden_dim
        net_param["out_dim"] = hidden_dim
        model = gnn_model("GatedGCN", net_param)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False,
                                  collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False,
                                collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False,
                                 collate_fn=dataset.collate)
        train_pred, train_labels = printPredictions(model, device, train_loader, "synthetic", 0, 4250)
        val_pred, val_labels = printPredictions(model, device, val_loader, "synthetic", 4250, 5000)
        train_pred.extend(val_pred)
        train_labels.extend(val_labels)
        print("\nThe Errors of the Small Synthetic is: ")
        print_errors(train_pred, train_labels)

        print("\nThe Errors of the JOB light ist: ")
        train_pred, train_labels = printPredictions(model, device, test_loader, "job-light", 0, 70)
        errors = print_errors(train_pred, train_labels)
        get_table_error_barchart(errors, DATASET_NAME)
        get_table_nr_preds_error_barchart(errors, DATASET_NAME)

    else:
        dataset = LoadData(DATASET_NAME)
        trainset, valset, testset = dataset.train, dataset.val, dataset.test


        net_param = get_net_params(device, dataset)
        net_param['hidden_dim'] = hidden_dim
        net_param["out_dim"] = hidden_dim
        model = gnn_model("GatedGCN", net_param)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False,
                                  collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False,
                                collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False,
                                 collate_fn=dataset.collate)

        train_pred, train_labels = printPredictions(model, device, train_loader, "synthetic", 0, 4250)
        val_pred, val_labels = printPredictions(model, device, val_loader, "synthetic", 4250, 5000)
        train_pred.extend(val_pred)
        train_labels.extend(val_labels)
        print("\nThe Errors of the Small Synthetic is: ")
        print_errors(train_pred, train_labels)

        print("\nThe Errors of the JOB light ist: ")
        train_pred, train_labels = printPredictions(model, device, test_loader, "job-light", 0, 70)
        print_errors(train_pred, train_labels)


#load_and_predict("out/Logical_regression/checkpoints/GatedGCN_Logical_GPU0_00h22m52s_on_Jun_07_2021/RUN_/epoch_196.pkl", "Logical", 128, 300)
load_and_predict("out/Logical-Samples_regression/checkpoints/GatedGCN_Logical-Samples_GPU0_00h46m16s_on_Jun_07_2021/RUN_/epoch_168.pkl", "Logical-Samples", 128, 300)
load_and_predict("out/Logical-Big_regression/checkpoints/GatedGCN_Logical-Big_GPU0_17h13m43s_on_Jun_07_2021/RUN_/epoch_168.pkl", "Logical-Big", 128, 300)
#load_and_predict("out/Logical_regression/checkpoints/GatedGCN_Logical_GPU0_21h23m42s_on_Jun_07_2021/RUN_/epoch_220.pkl", "Logical", 32, 300)
#load_and_predict("out/Logical-Samples_regression/checkpoints/GatedGCN_Logical-Samples_GPU0_22h16m59s_on_Jun_07_2021/RUN_/epoch_149.pkl", "Logical-Samples", 32, 300)
#load_and_predict("out/Logical-Big-Samples_regression/checkpoints/GatedGCN_Logical-Big-Samples_GPU0_23h13m34s_on_Jun_07_2021/RUN_/epoch_298.pkl", "Logical-Samples", 128, 300)
#load_and_predict("out/Logical-Big_regression/checkpoints/GatedGCN_Logical-Big_GPU0_13h02m46s_on_Jun_08_2021/RUN_/epoch_124.pkl", "Logical", 64, 300)
load_and_predict("out/Logical-Big-Samples_regression/checkpoints/GatedGCN_Logical-Big-Samples_GPU0_11h59m33s_on_Jun_09_2021/RUN_/epoch_238.pkl", "Logical-Big-Samples", 128, 200)


