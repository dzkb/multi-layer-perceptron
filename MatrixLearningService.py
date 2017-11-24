from threading import Thread
import numpy as np
import MatrixNet
import queue
import random
import time
from typing import *


class MatrixLearningService(Thread):
    def __init__(self, net: MatrixNet,producer_queue: queue.Queue,
                 consumer_queue: queue.Queue,
                 training_set,
                 validation_set,
                 learning_rate: float,
                 momentum_rate: float,
                 epoch_count=-1,
                 l2_λ=0,
                 dropout_chance=0):
        Thread.__init__(self)
        self.messages_out = producer_queue
        self.messages_in = consumer_queue
        self.net = net
        self.training_set = training_set
        self.validation_set = validation_set
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.max_epochs = epoch_count
        self.epoch_count = 0
        # some momentum-related stuff here
        # regularization paramters
        self.l2_λ = l2_λ
        self.dropout_chance = dropout_chance

    def run(self):
        if self.max_epochs is not -1:
            for i in range(self.max_epochs):
                if self.messages_in.empty():
                    print("epoch", self.epoch_count)
                    start = time.time()
                    mse_training = self.epoch_step()
                    print(time.time() - start)
                else:
                    break

            self.error(self.validation_set)


    def epoch_step(self):
        random.shuffle(self.training_set)
        errors = []
        correct_predictions = 0

        for p in range(len(self.training_set)):
            print("Sample {}/{}".format(p, len(self.training_set)))
            sample = self.training_set[p]
        # for sample in self.training_set:

            sample_input = sample[0]
            sample_label = sample[1]

            Δw = [np.zeros(layer_weights.shape) for layer_weights in self.net.weights]
            Δb = [np.zeros(layer_biases.shape) for layer_biases in self.net.biases]

            # 1 Feed-forward

            output = sample_input
            layers_outputs = []
            layers_nets = []
            layers_outputs.append(output)

            # 1.1 Calculate nets and activations for each layer without the last

            for w, b in zip(self.net.weights[:-1], self.net.biases[:-1]):
                net = np.dot(w, output) + b
                if self.dropout_chance > 0:
                    net *= np.random.binomial(1, self.dropout_chance, size=net.shape) / self.dropout_chance
                output = self.net.activation_f(net)
                layers_nets.append(net)
                layers_outputs.append(output)

            # 1.2 Calculate output layer

            net = np.dot(self.net.weights[-1], output) + self.net.biases[-1]
            output = self.net.output_activation_f(net)
            layers_nets.append(net)
            layers_outputs.append(output)

            # 2 Backpropagation
            # 2.1 Calculate errors
            # 2.1.1 Output layer
            # δ is the derivative of the cost function
            # δ =(Y - Y ̂)⨀f'(net)
            δ_w = (layers_outputs[-1] - sample_label) * self.net.output_activation_f.derivative(layers_nets[-1])
            Δb[-1] = δ_w
            Δw[-1] = np.dot(δ_w, layers_outputs[-2].transpose())

            # 2.1.2 Hidden layers
            for layer in range(2, self.net.num_layers):
                δ_w = np.dot(self.net.weights[-layer+1].transpose(), δ_w) \
                      * self.net.output_activation_f.derivative(layers_nets[-layer])
                Δb[-layer] = δ_w
                Δw[-layer] = np.dot(δ_w, layers_outputs[-layer - 1].transpose())

            # 2.2 Update weights

            self.net.weights = [((1 - self.l2_λ) * w - (self.learning_rate * δ_w)) for (w, δ_w) in zip(self.net.weights, Δw)]
            self.net.biases = [b - (self.learning_rate * δ_b) for (b, δ_b) in zip(self.net.biases, Δb)]

        self.epoch_count += 1
        return self.error(self.training_set)

    def error(self, test_set):
        inputs = np.hstack((input[0] for input in test_set))
        labels = np.hstack((input[1] for input in test_set))
        predictions = self.net.predict(inputs)
        print(np.sum((predictions - labels) ** 2).reshape(1)/len(test_set))


    def validate(self):
        pass

    def dump_shapes(self):
        print("-" * 10)
        for w, b in zip(self.net.weights, self.net.biases):
            print(w.shape, b.shape)