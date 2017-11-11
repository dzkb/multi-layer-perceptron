import tkinter as tk
import queue
from functions import Sigmoid, Linear
import LearningService
import Net
import datasets as data
import pickle
import time
import plotting
import os


class Application(tk.Frame):
    def __init__(self, current_net, master=None):
        super().__init__(master, width=300, height=300)
        self.net = current_net
        self.pack()
        self.grid(column=0, row=0, columnspan=2, rowspan=6)

        self.create_widgets()

    def stop_learning(self):
        consumer_queue.put("halt")

    def init_net(self):
        self.net.initialize_weights(len(training_set[0][0]), 10, 0.1)

    def start_learning(self):
        learning_service = LearningService.LearningService(
            self.net, producer_queue, consumer_queue, training_set, validation_set, 0.5, 0.25, 100)
        learning_service.start()

    def save_model(self):
        with open("models\\" + generate_filename(), "wb") as file:
            pickle.dump(self.net, file)
        print("Model saved")

    def load_model(self):
        filename = self.model_file.get()
        with open(filename, "rb") as model_file:
            self.net = pickle.load(model_file)
        print("Model loaded")

    def predict(self):
        test_directory = self.test_directory.get()
        test_data = data.load_test_set(test_directory)
        correct_predictions = 0
        for sample_record in test_data:
            sample = sample_record[0]
            label = sample_record[1]
            filename = sample_record[2]
            output = self.net.predict(sample)
            prediction = output.index(max(output))
            print("{}\tLabel:\t{}\tPrediction\t{}\tOutput:\t{}".format(filename, label, prediction, output))
            correct_predictions += 1 if prediction == int(label) else 0
        print("-" * 5, "Summary", "-" * 5)
        print("Correct predictions: {}/{}".format(correct_predictions, len(test_data)))
        print("Performance: {0:.2f}".format(correct_predictions/len(test_data)))


    def create_widgets(self):
        self.net_init = tk.Button(self, command=self.init_net)
        self.net_init["text"] = "Initialize net"
        self.net_init.grid(column=0, row=0, columnspan=2, sticky="we")
        # self.net_init.pack(side="top")

        self.start_learning = tk.Button(self, command=self.start_learning)
        self.start_learning["text"] = "Start learning"
        self.start_learning.grid(column=0, row=1, sticky="we")
        # self.start_learning.pack(side="top")

        self.stop_learning = tk.Button(self, command=self.stop_learning)
        self.stop_learning["text"] = "Stop learning"
        self.stop_learning.grid(column=1, row=1, sticky="we")
        # self.stop_learning.pack(side="top")

        self.save_model_button = tk.Button(self, command=self.save_model)
        self.save_model_button["text"] = "Save model"
        self.save_model_button.grid(column=0, row=2, columnspan=2, sticky="we")
        # self.save_model_button.pack(side="top")

        self.model_file = tk.StringVar()
        self.model_file_entry = tk.Entry(self, textvariable=self.model_file)
        # self.model_file_entry.pack(side="top")
        self.model_file_entry.grid(column=0, row=3)
        self.model_file.set("models\\val_0.08.pkl")

        self.load_model_button = tk.Button(self, command=self.load_model)
        self.load_model_button["text"] = "Load model"
        self.load_model_button.grid(column=1, row=3, sticky="we")
        # self.load_model_button.pack(side="top")

        self.test_directory = tk.StringVar()
        self.test_directory_entry = tk.Entry(self, textvariable=self.test_directory)
        # self.test_directory_entry.pack(side="left")
        self.test_directory_entry.grid(column=0, row=4)
        self.test_directory.set("data\\test")

        self.predict_button = tk.Button(self, command=self.predict)
        self.predict_button["text"] = "Predict"
        self.predict_button.grid(column=1, row=4, sticky="we")
        # self.predict_button.pack(side="right")


def generate_filename():
    return str(int(time.time())) + ".model.pkl"


def check_messages(input_queue: queue.Queue):
    if not input_queue.empty():
        training_results, validation_results = input_queue.get()
        plotting.plot(training_results, validation_results)
    root.after(100, check_messages, input_queue)

net = None

if __name__ == "__main__":
    producer_queue = queue.Queue()
    consumer_queue = queue.Queue()

    # dataset = data.load_training_set("data\\training_set")
    dataset = data.load_training_set("data\\augmented_set")
    training_set, validation_set = data.split_set(dataset, 0.8)

    activation_f = Sigmoid(-1)
    # activation_f = Linear()

    net = Net.Net(activation_f, 100)
    learning_service = None
    root = tk.Tk()
    app = Application(net, master=root)
    root.after(100, check_messages, producer_queue)
    app.mainloop()