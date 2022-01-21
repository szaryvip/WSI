from math import pi, sqrt, exp
import numpy as np
from typing import List, Tuple, Dict
from statistics import stdev


def read_data_from_txt(path: str, shuffle: bool = True) -> np.ndarray:
    data = []
    with open(path) as fp:
        for line in fp.readlines():
            line = line.split()
            correct_line = [float(value) for value in line]
            data.append(correct_line)
    data = np.array(data)
    np.random.shuffle(data)
    return data


def data_train_test(data: np.ndarray,
                    percent_train: float) -> List[np.ndarray]:
    border = int(len(data) * percent_train)
    return [data[:border], data[border:]]


def mean_of_numbers(numbers: np.ndarray) -> float:
    return sum(numbers)/len(numbers)


def stdev_of_numbers(numbers: np.ndarray) -> float:
    return stdev(numbers)


def separate_class(data: np.ndarray) -> dict:
    separated = {}
    for row in data:
        class_seed = row[-1]
        if class_seed not in separated:
            separated[class_seed] = list()
        separated[class_seed].append(row)
    return separated


def stats_for_data(data: np.ndarray) -> List[Tuple]:
    stats = [(mean_of_numbers(col), stdev_of_numbers(col),
              len(col)) for col in zip(*data)]
    del(stats[-1])
    return stats


def stats_for_class(data: np.ndarray) -> Dict:
    separated = separate_class(data)
    stats = {}
    for class_seed, datas in separated.items():
        stats[class_seed] = stats_for_data(datas)
    return stats


def normal_distribution_prob(x: float, mean: float, stdev: float) -> float:
    expon = exp(-((x - mean)**2 / (2*stdev**2)))
    prob = (1 / (sqrt(2*pi)*stdev)) * expon
    return prob


def class_probabilities(stats: Dict, row: np.ndarray) -> Dict:
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
    probas = class_probabilities(stats, row)
    return max(probas, key=probas.get)


def bayes(train: np.ndarray, test: np.ndarray) -> List[int]:
    stats = stats_for_class(train)
    predictions = []
    for data in test:
        predictions.append(predict(stats, data))
    return predictions


if __name__ == "__main__":
    data = read_data_from_txt("data/seeds_dataset.txt")
    train, test = data_train_test(data, 0.8)
    print(bayes(train, test))
