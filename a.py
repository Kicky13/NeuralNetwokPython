#!/usr/bin/pythonpyt

import NeuralNetwork
import sys
sys.path.insert(0, "/home/blegoh/.local/lib/python2.7/site-packages")
import json
import Dataset

datasets = [
    [[0.03, 0.05], [0.75]],
    [[0.05, 0.01], [0.82]],
    [[0.1, 0.02], [0.93]],
]
grandma = Dataset.Datasets.select()

# for p in grandma:
#    print p.kelas
nn = NeuralNetwork.NeuralNetwork(3, 1, 2)
# wh = [[0.84,0.58],[0.04,0.22],[0.13,0.06]]
wh = json.loads(sys.argv[1])
# print type(wh)
nn.set_hidden_layer_weight(wh)
# wo = [[0.72,0.73,0.75]]
wo = json.loads(sys.argv[2])
nn.set_output_layer_weight(wo)
for dataset in datasets:
    nn.add_dataset(dataset)

nn.trainingDataset()
