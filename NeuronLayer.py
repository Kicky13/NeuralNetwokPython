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
            print("bobot")
            print(self.neurons[i].weights)
            print("input")
            print(self.neurons[i].input)
            outputs.append(self.neurons[i].getOutput())
        return outputs
