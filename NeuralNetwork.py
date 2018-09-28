import NeuronLayer
import math


class NeuralNetwork:
    learningRate = 0.5

    def __init__(self, neuron_hidden_count, neuron_output_count, input_count):
        self.neuronHiddenCount = neuron_hidden_count
        self.neuronOutputCount = neuron_output_count
        self.inputCount = input_count
        self.hiddenLayers = NeuronLayer.NeuronLayer(neuron_hidden_count, 0.35)
        self.outputLayers = NeuronLayer.NeuronLayer(neuron_output_count, 0.6)
        self.dataset = []

    def add_dataset(self, data):
        self.dataset.append(data)

    def set_hidden_layer_weight(self, weight):
        self.hiddenLayers.set_weights(weight)

    def set_output_layer_weight(self, weight):
        self.outputLayers.set_weights(weight)

    def feed_forward(self, inputs):
        self.inputs = inputs
        y = self.hiddenLayers.feed_forward(inputs)
        return self.outputLayers.feed_forward(y)

    def get_error_total(self, target):
        self.target = target
        self.error = []
        for i in range(len(self.outputLayers.neurons)):
            self.error.append(self.outputLayers.neurons[i].calculate_error(target[i]))
        return sum(self.error)

    def derivative_output_to_hidden(self):
        # turunan error total thd output
        # -(target - output)
        dout = []
        for i in range(len(self.outputLayers.neurons)):
            dout.append(-1 * (self.target[i] - self.outputLayers.neurons[i].get_output()))

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

    def derivative_output_to_input(self):
        new_weight = []
        for i in range(len(self.hiddenLayers.neurons)):
            total = 0
            for j in range(len(self.outputLayers.neurons)):
                # turunan error thd output
                # -(target - output)
                dout = -1 * (self.target[j] - self.outputLayers.neurons[j].get_output())
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
            new_weight.append(a)
        return new_weight

    def train(self):
        w = self.derivative_output_to_hidden()
        for i in range(len(self.outputLayers.neurons)):
            self.outputLayers.neurons[i].weights = w[i]
        w = self.derivative_output_to_input()
        for i in range(len(self.hiddenLayers.neurons)):
            self.hiddenLayers.neurons[i].weights = w[i]

    def training_dataset(self):
        for i in range(100000):
            for dataset in self.dataset:
                self.feed_forward(dataset[0])
                self.get_error_total(dataset[1])
                self.train()
        for neuron_hidden in self.hiddenLayers.neurons:
            print neuron_hidden.weights
        print "a"
        for neuron_output in self.outputLayers.neurons:
            print neuron_output.weights

