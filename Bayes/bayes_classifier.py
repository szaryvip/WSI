from math import pi, sqrt, exp
import numpy as np
from typing import List, Tuple, Dict
from statistics import stdev


def read_data_from_txt(path: str, shuffle: bool = True) -> np.ndarray:
    """Reads data from txt file separated by [space] or [Tab]

    Args:
        path (str): path to file with data
        shuffle (bool, optional): shuffle data or not. Defaults to True.

    Returns:
        np.ndarray: data
    """
    data = []
    with open(path) as fp:
        for line in fp.readlines():
            line = line.split()
            correct_line = [float(value) for value in line]
            data.append(correct_line)
    data = np.array(data)
    if shuffle:
        np.random.shuffle(data)
    return data


def data_train_test(data: np.ndarray,
                    percent_train: float) -> List[np.ndarray]:
    """Divide data to train and test arrays

    Args:
        data (np.ndarray): data to divide
        percent_train (float): how many percent of data should be
                               used for training

    Returns:
        List[np.ndarray]: list with two arrays, train and test
    """
    border = int(len(data) * percent_train)
    return [data[:border], data[border:]]


def correct_classes(test: np.ndarray) -> np.ndarray:
    """Gets classes from test data

    Args:
        test (np.ndarray): test data

    Returns:
        np.ndarray: list of correct classes
    """
    correct = []
    for row in test:
        correct.append(row[-1])
    return np.array(correct)


def mean_of_numbers(numbers: np.ndarray) -> float:
    """Calculates mean of provided numbers

    Args:
        numbers (np.ndarray): numbers to calculate mean

    Returns:
        float: mean
    """
    return sum(numbers)/len(numbers)


def stdev_of_numbers(numbers: np.ndarray) -> float:
    """Calculates standard deviation of provided numbers

    Args:
        numbers (np.ndarray): numbers to calculate deviation

    Returns:
        float: standard deviation
    """
    return stdev(numbers)


def separate_class(data: np.ndarray) -> dict:
    """Divides data into classes

    Args:
        data (np.ndarray): data to divide

    Returns:
        dict: data divided in classes
    """
    separated = {}
    for row in data:
        class_seed = row[-1]
        if class_seed not in separated:
            separated[class_seed] = list()
        separated[class_seed].append(row)
    return separated


def stats_for_data(data: np.ndarray) -> List[Tuple]:
    """Calculates statistics for data

    Args:
        data (np.ndarray): data to calculate

    Returns:
        List[Tuple]: list of mean, standard deviation and length
        for each column in data
    """
    stats = [(mean_of_numbers(col), stdev_of_numbers(col),
              len(col)) for col in zip(*data)]
    del(stats[-1])
    return stats


def stats_for_class(data: np.ndarray) -> Dict:
    """Generates dictionary with statistics for each class

    Args:
        data (np.ndarray): data to calculate

    Returns:
        Dict: statistics for each class
    """
    separated = separate_class(data)
    stats = {}
    for class_seed, datas in separated.items():
        stats[class_seed] = stats_for_data(datas)
    return stats


def normal_distribution_prob(x: float, mean: float, stdev: float) -> float:
    """Calculates probability in normal distribution

    Args:
        x (float): argument
        mean (float): mean of data
        stdev (float): standard deviation of data

    Returns:
        float: probability
    """
    expon = exp(-((x - mean)**2 / (2*stdev**2)))
    prob = (1 / (sqrt(2*pi)*stdev)) * expon
    return prob


def class_probabilities(stats: Dict, row: np.ndarray) -> Dict:
    """Generates dictionary with probabilities for each class

    Args:
        stats (Dict): statistics of model
        row (np.ndarray): row of data

    Returns:
        Dict: probabilities for each class
    """
    total_rows = sum([stats[class_seed][0][2] for class_seed in stats])
    probas = {}
    for class_seed, class_stats in stats.items():
        probas[class_seed] = stats[class_seed][0][2]/float(total_rows)
        for index, i in enumerate(class_stats):
            mean, stdev = i[0], i[1]
            probas[class_seed] *= normal_distribution_prob(row[index],
                                                           mean, stdev)
    return probas


def predict(stats: Dict, row: np.ndarray) -> int:
    """Predicts class for row of data

    Args:
        stats (Dict): stats of model
        row (np.ndarray): data to predict

    Returns:
        int: prediction of class
    """
    probas = class_probabilities(stats, row)
    return max(probas, key=probas.get)


def bayes(train: np.ndarray, test: np.ndarray) -> List[int]:
    """Creates model from train data and predicts classes
    for each row in test data

    Args:
        train (np.ndarray): data to train model
        test (np.ndarray): data for test

    Returns:
        List[int]: predicted classes for test data
    """
    stats = stats_for_class(train)
    predictions = []
    for data in test:
        predictions.append(predict(stats, data))
    return predictions


if __name__ == "__main__":
    data = read_data_from_txt("data/seeds_dataset.txt", False)
    train, test = data_train_test(data, 0.8)
    print(bayes(train, test))
    print(correct_classes(test))
