import math


class Neuron:

    def __init__(self, bias):
        self.bias = bias

    def set_inputs(self, input):
        self.input = input

    def set_weights(self, weights):
        self.weights = weights

    def sum(self):
        jumlah = 0
        for i in range(len(self.input)):
            jumlah += self.input[i] * self.weights[i]
        return jumlah + self.bias

    @staticmethod
    def sigmoid(total):
        return 1 / (1 + math.exp(-total))

    def get_output(self):
        self.output = self.sigmoid(self.sum())
        return self.output

    def calculate_error(self, output_target):
        return 0.5 * (output_target - self.output) ** 2

