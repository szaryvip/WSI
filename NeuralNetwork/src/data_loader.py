import numpy as np
import idx2numpy
from copy import deepcopy


def normalization(data):
    """Normalizes data

    Args:
        data (List[float]): data to normalize

    Returns:
        List[float]: data after normalization
    """
    for row in data:
        max_value = np.amax(row[0])
        for i in range(len(row)-1):
            row[i] = row[i]/max_value
    return data


def load_data(path_data: str, path_labels: str) -> np.ndarray:
    """Loads data from idx1 and idx3 format

    Args:
        path_data (str): path to data
        path_labels (str): path to data's labels

    Returns:
        np.ndarray: normalized data with labels
    """
    data = idx2numpy.convert_from_file(path_data)
    data = deepcopy(data)
    labels = idx2numpy.convert_from_file(path_labels)
    labels = deepcopy(labels)
    data_with_label = []
    for image, label in zip(data, labels):
        # it takes more than 15 seconds for data to load when using lists
        image = image.flatten()
        data_with_label.append([image, label])
    data_with_label = normalization(data_with_label)
    np.random.shuffle(data_with_label)
    return data_with_label
