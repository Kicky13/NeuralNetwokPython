import NeuronLayer
import math


class NeuralNetwork:
    learningRate = 0.5

    def __init__(self, neuronHiddenCount, neuronOutputCount, inputCount):
        self.neuronHiddenCount = neuronHiddenCount
        self.neuronOutputCount = neuronOutputCount
        self.inputCount = inputCount
        self.hiddenLayers = NeuronLayer.NeuronLayer(neuronHiddenCount, 0.35)
        self.outputLayers = NeuronLayer.NeuronLayer(neuronOutputCount, 0.6)
        self.dataset = []

    def addDataset(self, data):
        self.dataset.append(data)

    def setHiddenLayerWeight(self, weight):
        self.hiddenLayers.setWeights(weight)

    def setOutputLayerWeight(self, weight):
        self.outputLayers.setWeights(weight)

    def feedForward(self, inputs):
        self.inputs = inputs
        y = self.hiddenLayers.feedForward(inputs)
        return self.outputLayers.feedForward(y)

    def getErrorTotal(self, target):
        self.target = target
        self.error = []
        for i in range(len(self.outputLayers.neurons)):
            self.error.append(self.outputLayers.neurons[i].calculateError(target[i]))
        return sum(self.error)

    def derivativeOutputToHidden(self):
        # turunan error total thd output
        # -(target - output)
        dout = []
        for i in range(len(self.outputLayers.neurons)):
            dout.append(-1 * (self.target[i] - self.outputLayers.neurons[i].getOutput()))

        # turunan output thd sum
        dsum = []
        for i in range(len(self.outputLayers.neurons)):
            dsum.append(math.exp(-self.outputLayers.neurons[i].sum()) / (
                    (1 + math.exp(-self.outputLayers.neurons[i].sum())) ** 2))

        # turunan sum thd input
        dweight = []
        for i in range(len(self.outputLayers.neurons)):
            dweight.append(self.outputLayers.neurons[i].input)

        # turunan error total thd input
        # sekaligus update weight
        a = []
        for i in range(len(self.outputLayers.neurons)):
            b = []
            for j in range(len(self.outputLayers.neurons[i].input)):
                turunan = dout[i] * dsum[i] * dweight[i][j]
                # new weight
                b.append(self.outputLayers.neurons[i].weights[j] - self.learningRate * turunan)
            a.append(b)
        return a

    def derivativeOutputToInput(self):
        newWeight = []
        for i in range(len(self.hiddenLayers.neurons)):
            total = 0
            for j in range(len(self.outputLayers.neurons)):
                # turunan error thd output
                # -(target - output)
                dout = -1 * (self.target[j] - self.outputLayers.neurons[j].getOutput())
                # turunan output thd net
                dsum = math.exp(-self.outputLayers.neurons[j].sum()) / (
                            (1 + math.exp(-self.outputLayers.neurons[j].sum())) ** 2)
                # turunan net thd input dr neuron hidden
                dinputh = self.outputLayers.neurons[j].weights[i]
                total += dout * dsum * dinputh
            # turunan output thd summing junction
            dsum = math.exp(-self.hiddenLayers.neurons[i].sum()) / ((1 + math.exp(-self.hiddenLayers.neurons[i].sum())) ** 2)
            a = []
            for k in range(len(self.hiddenLayers.neurons[i].weights)):
                turunan = dsum * total * self.inputs[k]
                a.append(self.hiddenLayers.neurons[i].weights[k] - self.learningRate * turunan)
            newWeight.append(a)
        return newWeight

    def train(self):
        w = self.derivativeOutputToHidden()
        for i in range(len(self.outputLayers.neurons)):
            self.outputLayers.neurons[i].weights = w[i]
        w = self.derivativeOutputToInput()
        for i in range(len(self.hiddenLayers.neurons)):
            self.hiddenLayers.neurons[i].weights = w[i]

    def trainingDataset(self):
        for i in range(100000):
            for dataset in self.dataset:
                self.feedForward(dataset[0])
                self.getErrorTotal(dataset[1])
                self.train()
        for neuron_hidden in  self.hiddenLayers.neurons:
            print neuron_hidden.weights
        print "a"
        for neuron_output in  self.outputLayers.neurons:
            print neuron_output.weights

