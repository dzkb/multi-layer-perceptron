import tkinter as tk
import queue
from functions import Sigmoid, Linear
import LearningService
import Net
import datasets as data
import pickle
import time

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def stop_learning(self):
        consumer_queue.put("halt")

    def init_net(self):
        net.initialize_weights(len(training_set[0][0]), 10, 0.1)

    def start_learning(self):
        learning_service = LearningService.LearningService(net, producer_queue, consumer_queue, training_set, validation_set, 0.15, False, 50)
        learning_service.start()

    def save_model(self):
        with open("models\\" + generate_filename(), "wb") as file:
            pickle.dump(net, file)

    def create_widgets(self):
        self.net_init = tk.Button(self, command=self.init_net)
        self.net_init["text"] = "Initialize net"
        self.net_init.pack(side="top")

        self.start_learning = tk.Button(self, command=self.start_learning)
        self.start_learning["text"] = "Start learning"
        self.start_learning.pack(side="top")

        self.stop_learning = tk.Button(self, command=self.stop_learning)
        self.stop_learning["text"] = "Stop learning"
        self.stop_learning.pack(side="top")

        self.save_model_button = tk.Button(self, command=self.save_model)
        self.save_model_button["text"] = "Save model"
        self.save_model_button.pack(side="right")

def generate_filename():
    return str(int(time.time())) + ".model.pkl"


def check_messages(input_queue: queue.Queue):
    if not input_queue.empty():
        print(input_queue.get())
    root.after(100, check_messages, input_queue)

if __name__ == "__main__":
    producer_queue = queue.Queue()
    consumer_queue = queue.Queue()

    dataset = data.load_training_set("data\\training_set")
    training_set, validation_set = data.split_set(dataset, 0.9)

    activation_f = Sigmoid(-3)
    # activation_f = Linear()

    net = Net.Net(activation_f, 40)
    learning_service = None
    root = tk.Tk()
    app = Application(master=root)
    root.after(100, check_messages, producer_queue)
    app.mainloop()