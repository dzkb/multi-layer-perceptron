from threading import Thread
import queue
import time
import Net
from random import shuffle
from typing import List, Tuple

class LearningService(Thread):
    def __init__(self, net: Net,producer_queue: queue.Queue,
                 consumer_queue: queue.Queue,
                 training_set: List[Tuple[Tuple[float], Tuple[float]]],
                 validation_set: List[Tuple[Tuple[float], Tuple[float]]],
                 learning_rate: float,
                 has_momentum: bool,
                 epoch_count=-1):
        Thread.__init__(self)
        self.messages_out = producer_queue
        self.messages_in = consumer_queue
        self.net = net
        self.training_set = training_set
        self.validation_set = validation_set
        self.learning_rate = learning_rate
        self.has_momentum = has_momentum
        self.max_epochs = epoch_count
        self.epoch_count = 0

    def run(self):
        if self.max_epochs is not -1:
            errors = []
            for i in range(self.max_epochs):
                if self.messages_in.empty():
                    mse = self.epoch_step()
                    print("epoch: {}".format(i))
                    print(mse)
                    errors.append(mse)
                    print(self.validate())
            print(errors)
        else:
            pass

    def epoch_step(self):
        shuffle(self.training_set)
        errors = []
        correct_predictions = 0
        for sample in self.training_set:
            sample_input = sample[0]
            biased_sample_input = sample_input + (1,)
            sample_label = sample[1]

            # 1 - Feed forward
            # 1.1 - Calculate net in hidden layer
            net_hidden = self.net.net_hidden(sample_input)

            # 1.2 - Calculate hidden layer output
            hidden_layer = [self.net.activation_f(net_j) for net_j in net_hidden] + [1]  # bias

            # 1.3 - Calculate net in output layer
            net_output = self.net.net_output(hidden_layer)

            # 1.4 - Calculate output layer
            output = [self.net.activation_f(output_k) for output_k in net_output]

            # 2 - Backpropagation
            # 2.1 - Calculate output error (output delta)
            δ_outputs = []
            for k in range(len(self.net.weights_output)):
                δ_pk = sample_label[k] - output[k]
                δ_outputs.append(δ_pk * self.net.activation_f.derivative(net_output[k]))
            # print(δ_outputs)

            # 2.2 - Calculate hidden layer error (hidden delta)
            δ_hiddens = []
            for j in range(len(self.net.weights_hidden)):
                weights_output_j = [weights_output[j] for weights_output in self.net.weights_output]
                δ_pj = sum(δ_pk * w_kj for (δ_pk, w_kj) in zip(δ_outputs, weights_output_j))
                δ_pj = δ_pj * self.net.activation_f.derivative(net_hidden[j])
                δ_hiddens.append(δ_pj)

            # 2.3 - Update output weights
            for k in range(len(self.net.weights_output)):
                for j in range(len(self.net.weights_output[k])):
                    self.net.weights_output[k][j] += self.learning_rate * δ_outputs[k] * hidden_layer[j]

            # 2.4 - Update hidden weights
            for j in range(len(self.net.weights_hidden)):
                for i in range(len(self.net.weights_hidden[j])):
                    self.net.weights_hidden[j][i] += self.learning_rate * δ_hiddens[j] * biased_sample_input[i]

            # 3 - Error calculation
            # 3.1 - Calculate output
            output = self.net.predict(sample_input)
            # print(sample_label, output)
            if output.index(max(output)) == sample_label.index(max(sample_label)):
                correct_predictions += 1

            e_p = 1/2 * sum((label - neuron_output) ** 2 for (label, neuron_output) in zip(sample_label, output))
            errors.append(e_p)
        return sum(errors) / len(errors)

    def validate(self):
        error = 0
        correct_predictions = 0
        for sample in self.validation_set:
            sample_input = sample[0]
            sample_labels = sample[1]
            outputs = self.net.predict(sample_input)
            e_p = 1 / 2 * sum((label - neuron_output) ** 2 for (label, neuron_output) in zip(sample_labels, outputs))
            error += e_p
            correct_predictions += 1 if outputs.index(max(outputs)) == sample_labels.index(max(sample_labels)) else 0
        error = error/len(self.validation_set)
        return error, correct_predictions, len(self.validation_set)
