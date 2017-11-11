from PIL import Image
import os
import random
from math import floor


def to_categorical(value):
    encoded = [0 for _ in range(10)]
    encoded[value] = 1
    return tuple(encoded)


def binarize(image):
    return tuple(1 - (x/255) for x in iter(image))


def load_single_test(filename):
    with open(filename, "rb") as file:
        image = Image.open(file)
        image = binarize(image)
        return image


def load_test_set(data_dir):
    dataset = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            label = filename[0]  # first characters indicates label
            with open(data_dir + "\\" + filename, "rb") as file:
                image = Image.open(file)
                print(filename, list(image.getdata()))
                image = image.convert("L")
                image = binarize(image.getdata())
                data_record = (image, label, filename)
                dataset.append(data_record)
    return dataset


def load_training_set(data_dir):
    dataset = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            label = filename[0]  # first characters indicates label
            with open(data_dir + "\\" + filename, "rb") as file:
                image = Image.open(file)
                print(filename, list(image.getdata()))
                image = image.convert("L")
                image = binarize(image.getdata())
                label = to_categorical(int(label))
                data_record = (image, label, filename)
                dataset.append(data_record)
    print("Loaded {} files".format(len(dataset)))
    return dataset


def split_set(dataset, training_size=0.7):
    random.shuffle(dataset)
    return dataset[:floor(len(dataset) * training_size)], dataset[floor(len(dataset) * training_size):]

if __name__ == '__main__':
    a = load_training_set("data\\training_set")
    print(a)
