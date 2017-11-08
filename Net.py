import random
from typing import *


class Net:

    def __init__(self, activation_function: Callable, neuron_count=100):
        self.neuron_count = neuron_count
        self.activation_f = activation_function
        self.weights_hidden = []
        self.weights_output = []

    def initialize_weights(self, n_inputs, n_outputs, threshold=1):
        self.weights_hidden = []
        self.weights_output = []

        for i in range(self.neuron_count):
            n_biased_inputs = n_inputs + 1
            neuron_weights = [random.uniform(-threshold, threshold) for _ in range(n_biased_inputs)]
            self.weights_hidden.append(neuron_weights)

        for i in range(n_outputs):
            n_biased_hiddens = self.neuron_count + 1
            neuron_weights = [random.uniform(-threshold, threshold) for _ in range(n_biased_hiddens)]
            self.weights_output.append(neuron_weights)

    def predict(self, input_sample):
        hidden_net = self.net_hidden(input_sample)
        hidden_layer = [self.activation_f(net_j) for net_j in hidden_net] + [1]

        outputs_net = self.net_output(hidden_layer)
        outputs = [self.activation_f(net_k) for net_k in outputs_net]

        return outputs

    def net_hidden(self, input_sample):
        biased_input = input_sample + (1,)
        hidden_layer = []
        for j in range(self.neuron_count):
            weights_j = self.weights_hidden[j]
            net_j = sum(x_p * w_p for (x_p, w_p) in zip(biased_input, weights_j))
            hidden_layer.append(net_j)
        return hidden_layer

    def net_output(self, hidden_layer):
        # input should be already biased
        biased_hidden_layer = hidden_layer #  + [1]
        output_net = []
        for k in range(len(self.weights_output)):
            weights_k = self.weights_output[k]
            net_k = sum(i_k * w_k for (i_k, w_k) in zip(biased_hidden_layer, weights_k))
            output_net.append(net_k)
        return output_net
