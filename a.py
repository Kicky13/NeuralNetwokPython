import NeuralNetwork

datasets = [
    [[0.03, 0.05], [0.75]],
    [[0.05, 0.01], [0.82]],
    [[0.1, 0.02], [0.93]],
]
nn = NeuralNetwork.NeuralNetwork(3, 1, 2)
nn.setHiddenLayerWeight([[0.84,0.58],[0.04,0.22],[0.13,0.06]])
nn.setOutputLayerWeight([[0.72,0.73,0.75]])
for dataset in datasets:
    nn.addDataset(dataset)

nn.trainingDataset()