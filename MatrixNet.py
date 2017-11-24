import numpy as np
import random


class MatrixNet():
    def __init__(self, activation_function, output_activation):
        self.layer_sizes = None
        self.num_layers = None
        self.layers = None
        self.biases = None
        self.weights = None
        self.activation_f = activation_function
        self.output_activation_f = output_activation

    def initialize_weights(self, layers, threshold=1):
        self.num_layers = len(layers)
        self.layer_sizes = layers
        self.biases = [np.random.uniform(-threshold, threshold, size=(y, 1)) for y in layers[1:]]
        self.weights = [np.random.uniform(-threshold, threshold, size=(y, x)) for (x, y) in zip(layers[:-1], layers[1:])]

    def predict(self, input_sample):
        output = input_sample

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            output = self.activation_f(np.dot(w, output) + b)

        # Allow different output activation function (e.g. for softmax)
        output = self.output_activation_f(np.dot(self.weights[-1], output) + self.biases[-1])

        return output
