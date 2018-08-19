import NeuralNetwork
import Neuron
import NeuronLayer

a = NeuralNetwork.NeuralNetwork(2,2,2)
a.setHiddenLayerWeight([[0.15,0.2],[0.25,0.3]])
a.setOutputLayerWeight([[0.4,0.45],[0.5,0.55]])
a.feedForward([0.05,0.1])
a.getErrorTotal([0.01,0.99])
a.derivativeOutputToHidden()
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