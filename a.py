import NeuralNetwork
import Neuron
import NeuronLayer

# a = NeuralNetwork.NeuralNetwork(2, 2, 2)
# a.setHiddenLayerWeight([[0.15, 0.2], [0.25, 0.3]])
# a.setOutputLayerWeight([[0.4, 0.45], [0.5, 0.55]])
# a.feedForward([0.05, 0.1])
# a.getErrorTotal([0.01, 0.99])
# a.derivativeOutputToHidden()
# a.derivativeOutputToInput()
# b = Neuron.Neuron(0.35)
# b.setWeights([0.15,0.2])
# b.setInputs([0.05,0.1])
# print(b.weights)
# print(b.sum())
#
# c = NeuronLayer.NeuronLayer(2,0.35)
# c.setWeights([[0.15,0.2],[0.25,0.3]])
# print(c.feedForward([0.05,0.1]))


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