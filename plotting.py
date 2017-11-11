import matplotlib.pyplot as plt


def plot(training, val):
    plt.figure(1)
    tr_plot = plt.plot(training, label="Training set")
    val_plot = plt.plot(val, label="Validation set")
    plt.legend(["Training set", "Validation set"])
    plt.title("Mean Squared Error during subsequent epochs of training")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.show()