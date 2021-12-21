from neural_network import NeuralNetwork
from data_loader import load_data
import numpy as np
import os
import random
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from playground import show_confusion_matrix

import matplotlib.pyplot as plt

DATA_PERCENTAGE = 0.02

# TODO:
# data analysis

# neurons number
# layers number
# learning rate
# epochs

# example network
# example classifications

# error matrix
# quality metrics

def show_matrix(predictions: np.ndarray,
                          expectations: np.ndarray):
    class_list = [i for i in range(10)]
    cm = confusion_matrix(expectations, predictions, labels=class_list)
    show_confusion_matrix(cm, class_list)


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
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1 Score': None
    }

    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['Precision'] = tp / (tp + fp)
    metrics['Recall'] = tp / (tp + fn)
    metrics['F1 Score'] = 2*tp / (2*tp + fp + fn)

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
    metrics = get_metrics(confusion_matrix)

    # print(confusion_matrix)
    # print(metrics)
    # print(predictions)
    # print(expectations)

    return confusion_matrix, metrics


def plot(dict):
    for key in dict:
        if isinstance(dict[key], list):
            changing_parameter = key
            changing_parameter_values = dict[key]
            values_number = len(dict[key])
    for key in dict:
        if key != changing_parameter:
            dict[key] = [
                dict[key]
                for _ in range(values_number)
            ]

    confusion_matrix_list = []
    metrics_list = []
    times_list = []

    all_tests_start_time = time.time()

    for index in range(values_number):

        # os.system('clear')
        print(f'Test: {index+1}/{values_number}')

        start_time = time.time()

        confusion_matrix, metrics = test(
            dict['hidden_layers_number'][index],
            dict['neurons_in_layer_number'][index],
            dict['epochs_number'][index],
            dict['learning_rate'][index]
        )

        times_list.append(time.time() - start_time)
        confusion_matrix_list.append(confusion_matrix)
        metrics_list.append(metrics)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if changing_parameter == 'hidden_layers_number':
        ax1.set_xlabel('Hidden layers')
    elif changing_parameter == 'neurons_in_layer_number':
        ax1.set_xlabel('Neurons in layer')
    elif changing_parameter == 'epochs number':
        ax1.set_xlabel('Epochs')
    elif changing_parameter == 'learning rate':
        ax1.set_xlabel('Learning rate')

    for metric in metrics_list[0]:
        if metric in ('Accuracy', 'Precision'):
            values = [
                metrics[metric]
                for metrics in metrics_list
            ]
            ax1.plot(changing_parameter_values, values, marker='o', label=metric)

    ax1.set_ylabel('Metrics')
    ax1.set_ylim((0, 1))
    ax1.legend(
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15)
    )

    ax2.plot(changing_parameter_values, times_list, color='red', marker='o')
    ax2.set_ylabel('Time', color='red')
    plt.xticks(changing_parameter_values, changing_parameter_values)

    plt.tight_layout()

    plt.savefig(f'plots/{changing_parameter}.png')
    # plt.show()

    print(f'All tests time: {round(time.time() - all_tests_start_time)}s\n')


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

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

    # print(training_data_last_index)
    # print(test_data_last_index)

    # confusion_matrix, metrics = test(
    #     hidden_layers_number=2,
    #     neurons_in_layer_number=10,
    #     epochs_number=10,
    #     learning_rate=1
    # )

    # # print(confusion_matrix)
    # print(metrics)

    # quit()

    plot(
        {
            'hidden_layers_number': 2,
            'neurons_in_layer_number': list(range(5, 30, 5)),
            'epochs_number': 10,
            'learning_rate': 1
        }
    )

    plot(
        {
            'hidden_layers_number': list(range(1, 5, 1)),
            'neurons_in_layer_number': 10,
            'epochs_number': 10,
            'learning_rate': 1
        }
    )

    plot(
        {
            'hidden_layers_number': 2,
            'neurons_in_layer_number': 10,
            'epochs_number': list(range(1, 20, 1)),
            'learning_rate': 1
        }
    )

    plot(
        {
            'hidden_layers_number': 2,
            'neurons_in_layer_number': 10,
            'epochs_number': 10,
            'learning_rate': np.arange(0.5, 5.5, 0.5).tolist()
        }
    )
