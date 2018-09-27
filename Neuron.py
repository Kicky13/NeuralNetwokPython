import math


class Neuron:

    def __init__(self, bias):
        self.bias = bias

    def setInputs(self, input):
        self.input = input

    def setWeights(self, weights):
        self.weights = weights

    def sum(self):
        jumlah = 0
        for i in range(len(self.input)):
            jumlah += self.input[i] * self.weights[i]
        return jumlah + self.bias

    def sigmoid(self, total):
        return 1 / (1 + math.exp(-total))

    def getOutput(self):
        self.output = self.sigmoid(self.sum())
        return self.output

    def calculateError(self, ouputTarget):
        return 0.5 * (ouputTarget - self.output) ** 2

