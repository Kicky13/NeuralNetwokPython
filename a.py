#!/usr/bin/pythonpyt
import pymysql.cursors
from NeuralNetwork import NeuralNetwork
import sys
import json
from Dataset import Datasets

data_set = Datasets.select()

nn = NeuralNetwork(4, 1, 4)

wh = json.loads(sys.argv[1])
nn.set_hidden_layer_weight(wh)

wo = json.loads(sys.argv[2])
nn.set_output_layer_weight(wo)

for data in data_set:
    inputs = [float(data.asal_smp), int(data.lebih_tua), float(data.jurusan), float(data.pekerjaan_ortu)]
    if data.kelas == 'nakal':
        output = [1]
    else:
        output = [0]
    nn.add_dataset([inputs, output])

nn.training_dataset()
