import csv
import json

pos_enc = [True, False]
hidden_dim = [100, 200, 300, 400]
batch_size = [32, 64, 128, 256]
layers = [4, 8, 16, 32]

with open("config_0.json") as j_file:
    config = json.load(j_file)

file_names = []
for pe in pos_enc:
    for hd in hidden_dim:
        for bs in batch_size:
            for l in layers:
                config["net_params"]["pos_enc"] = pe
                config["net_params"]["hidden_dim"] = hd
                config["net_params"]["out_dim"] = hd
                config["params"]["batch_size"] = bs
                config["net_params"]["L"] = l
                file_name = "config_pos_enc_{}_hidden_dim_{}_batch_size_{}_layers_{}.json".format(pe, hd, bs, l)
                file_names.append(file_name)
                with open(file_name, "w") as out_file:
                    json.dump(config, out_file, indent=2)

with open("../../scripts/script_configurations.sh", "w") as sh_file:
    sh_file.write("#!/bin/bash\n")
    for conf_file in file_names:
        sh_file.write("python JOBSynthetic_Graph_Regression.py --config \'../../configs/JOBSynthetic/{}\'\n".format(conf_file))
