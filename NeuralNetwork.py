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
