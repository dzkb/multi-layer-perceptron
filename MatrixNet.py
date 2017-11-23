import numpy as np
import random


class MatrixNet():
    def __init__(self, activation_function, neuron_count=100):
        self.neuron_count = neuron_count
        self.activation_f = activation_function
        self.output_activation_f = None
        self.weights_hidden = None
        self.weights_output = None

    def initialize_weights(self, n_inputs, n_outputs, threshold=1):
        n_biased_inputs = n_inputs + 1
        n_biased_hiddens = self.neuron_count + 1

        self.weights_hidden = np.zeros((self.neuron_count, n_biased_inputs))
        self.weights_output = np.zeros((n_outputs, n_biased_hiddens))

        for i in range(self.neuron_count):
            for j in range(n_biased_inputs):
                self.weights_hidden[i][j] = random.uniform(-threshold, threshold)
            # neuron_weights = [random.uniform(-threshold, threshold) for _ in range(n_biased_inputs)]
            # self.weights_hidden.append(neuron_weights)

        for i in range(n_outputs):
            for j in range(n_biased_hiddens):
                self.weights_output[i][j] = random.uniform(-threshold, threshold)
            # neuron_weights = [random.uniform(-threshold, threshold) for _ in range(n_biased_hiddens)]
            # self.weights_output.append(neuron_weights)


    def predict(self, input_sample):
        pass