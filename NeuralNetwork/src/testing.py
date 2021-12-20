from neural_network import NeuralNetwork
from data_loader import load_data
import numpy as np
import random

import matplotlib as plt

DATA_PERCENTAGE = 1

# TODO:
# data analysis
# neurons number
# lauers number
# learning rate
# epochs
# example network
# example classifications
# error matrix
# quality metrics


def create_confusion_matrix(
    predictions: np.ndarray,
    expectations: np.ndarray
) -> dict:
    confusion_matrix = {
        key: {
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
        for key in range(10)
    }

    for digit in confusion_matrix:
        for predicition, expectation in zip(predictions, expectations):
            if expectation == digit and predicition == digit:
                confusion_matrix[digit]['tp'] += 1
            elif expectation != digit and predicition != digit:
                confusion_matrix[digit]['tn'] += 1
            elif expectation != digit and predicition == digit:
                confusion_matrix[digit]['fp'] += 1
            elif expectation == digit and predicition != digit:
                confusion_matrix[digit]['fn'] += 1

    '''
    --------------------------------------------------------
    different approach, not sure if correct, want to test later
    V---V---V---V---V---V---V---V---V---V---V---V---V---V---V
    '''
    # for predicition, expectation in zip(predictions, expectations):
    #     # correct prediciton
    #     if predicition == expectation:
    #         confusion_matrix[predicition]['tp'] += 1
    #         for key in list(confusion_matrix.keys()):
    #             if key != predicition:
    #                 confusion_matrix[key]['tn'] += 1
    #     # incorrect prediction
    #     else:
    #         confusion_matrix[predicition]['fp'] += 1
    #         confusion_matrix[expectation]['fn'] += 1
    #         for key in confusion_matrix:
    #             if key != predicition and key != expectation:
    #                 confusion_matrix[key]['tn'] += 1

    return confusion_matrix


def get_metrics(confusion_matrix: dict) -> dict:
    tp, tn, fp, fn = 0, 0, 0, 0

    for key in confusion_matrix:
        tp += confusion_matrix[key]['tp']
        tn += confusion_matrix[key]['tn']
        fp += confusion_matrix[key]['fp']
        fn += confusion_matrix[key]['fn']

    metrics = {
        'accuracy': None,
        'precision': None
    }

    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp)

    return metrics


def test(
    hidden_layers_number: int,
    neurons_in_layer_number: int,
    epochs_number: int,
    learning_rate: float = 1
):
    Chad = NeuralNetwork(
        hidden_layers_number,
        neurons_in_layer_number,
        epochs_number
    )

    predictions = Chad.back_propagation(
        training_data,
        test_data,
        learning_rate
    )

    expectations = [row[1] for row in test_data]

    confusion_matrix = create_confusion_matrix(predictions, expectations)

    # print(confusion_matrix)
    print(
        get_metrics(confusion_matrix)
    )
    # print(predictions)
    # print(expectations)


if __name__ == "__main__":
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

    # print(training_data_last_index)

    training_data = training_data[:training_data_last_index]
    test_data = test_data[:test_data_last_index]

    test(
        hidden_layers_number=2,
        neurons_in_layer_number=10,
        epochs_number=10,
        learning_rate=1
    )
