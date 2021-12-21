from neural_network import NeuralNetwork
from data_loader import load_data

import random

import numpy as np
import matplotlib.pyplot as plt

DATA_PERCENTAGE = 10

class_names = [i for i in range(10)]

def plot_image(image, test_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel(
        f'Label: {test_label}',
        color = 'green'
    )

def plot_value_array(predictions, test_label):
    # plt.grid(True)
    plt.xticks(range(10))
    # plt.yticks([])
    predictions = list(predictions)
    plot = plt.bar(range(10), predictions, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions)
    # pred.append(predicted_label)
    plot[predicted_label].set_color('blue')
    plot[test_label].set_color('green')


def show_example(
    training_data,
    test_data,
    hidden_layers_number=2,
    neurons_in_layer_number=15,
    epochs_number=10,
    learning_rate=0.1
):
    Chad = NeuralNetwork(
        hidden_layers_number,
        neurons_in_layer_number,
        epochs_number,
        raw_output=True
    )

    predictions_list = Chad.back_propagation(
        training_data,
        test_data,
        learning_rate
    )

    predictions = [
        np.argmax(predictions)
        for predictions in predictions_list
    ]

    expectations = [row[1] for row in test_data]


    good = None
    bad = None
    i=0

    for prediciton, expectation in zip(
        predictions,
        expectations
    ):
        if good is None and prediciton == expectation:
            good = i
        elif bad is None and prediciton != expectation:
            bad = i
        i+=1

    # print(predictions)
    # print(expectations)
    # print(good)
    # print(bad)

    # good

    test_image_flat = test_data[good][0]
    test_image = []

    # divide flat image into rows of pixels
    for row in range(28):
        pixels = test_image_flat[row*28:row*28+28]
        for column in range(28):
            pixels[column] *= 255 
        test_image.append(pixels)

    plt.figure(figsize=(6,6))

    plt.subplot(2,2,1)
    plot_image(test_image, test_data[good][1])
    
    plt.subplot(2,2,2)
    plot_value_array(predictions_list[good], test_data[good][1])

    # bad

    test_image_flat = test_data[bad][0]
    test_image = []

    # divide flat image into rows of pixels
    for row in range(28):
        pixels = test_image_flat[row*28:row*28+28]
        for column in range(28):
            pixels[column] *= 255 
        test_image.append(pixels)

    plt.subplot(2,2,3)
    plot_image(test_image, test_data[bad][1])

    plt.subplot(2,2,4)
    plot_value_array(predictions_list[bad], test_data[bad][1])
    
    plt.savefig(f'plots/example.png')
    plt.show()

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    training_data = load_data(
        'data/train-images.idx3-ubyte',
        'data/train-labels.idx1-ubyte'
    )
    test_data = load_data(
        'data/t10k-images.idx3-ubyte',
        'data/t10k-labels.idx1-ubyte'
    )

    training_data_last_index = round(
        len(training_data) * DATA_PERCENTAGE/100
    )
    test_data_last_index = round(
        len(test_data) * DATA_PERCENTAGE/100
    )
    training_data = training_data[:training_data_last_index]
    test_data = test_data[:test_data_last_index]

    print(training_data_last_index)

    show_example(training_data, test_data)

    