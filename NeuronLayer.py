import Neuron


class NeuronLayer:

    def __init__(self, neuroncount=1, bias=0):
        self.neurons = []
        for i in range(neuroncount):
            self.neurons.append(Neuron.Neuron(bias))

    def setWeights(self, weights):
        for i in range(len(weights)):
            self.neurons[i].weights = weights[i]

    def feedForward(self, inputs):
        outputs = []
        for i in range(len(self.neurons)):
            self.neurons[i].setInputs(inputs)
            outputs.append(self.neurons[i].getOutput())
        return outputs
