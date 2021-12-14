import csv
import random
import matplotlib.pyplot as plt
import numpy as np


def read_from_csv(path):
    """Read data from csv and save it in dict

    Args:
        path (str): path to csv file

    Returns:
        dict: dictionary with data
    """
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_row = {'buying': row[0],
                        'maint': row[1],
                        'doors': row[2],
                        'persons': row[3],
                        'lug_boot': row[4],
                        'safety': row[5],
                        'class': row[6]}
            data.append(data_row)
    return data


def train_test_data(data, percent_to_train, mix):
    """Divide data to train and test sets

    Args:
        data (dict): all data to divide
        percent_to_train (float): percent of train data
        mix (bool): true if u want to mix data or false
                    if u want to get it in original order

    Returns:
        tuple: train_data, test_data
    """
    if mix:
        random.shuffle(data)
    border = int(len(data) * percent_to_train)
    train_data = data[:border]
    test_data = data[border:]
    return train_data, test_data


def cross_validation(data, times):
    """Divide train data to train and validation data

    Args:
        train_data (dict): data to divide
        times (int): one portion from times portions will be validate data

    Returns:
        list: list of train and validation datas
    """
    border = len(data)//times
    datas = []
    if times == 1:
        return [[data, data]]
    for i in range(times-1):
        validation_data = data[i*border:(i+1)*border]
        data = data[:i*border] + data[(i+1)*border:]
        datas.append([data, validation_data])
    return datas


if __name__ == "__main__":
    data = read_from_csv("data/car.data")
    atr_labels = ["vhigh","high", "med", "low"]
    vgood = [0,0,0,0]
    good = [0,0,0,0]
    acc = [0,0,0,0]
    unacc = [0,0,0,0]
    x = np.arange(len(atr_labels))
    width = 0.20
    for row in data:
        for index, atr in enumerate(atr_labels):
            if row["buying"] == atr:
                if row["class"] == "vgood":
                    vgood[index] += 1
                if row["class"] == "good":
                    good[index] += 1
                if row["class"] == "acc":
                    acc[index] += 1
                if row["class"] == "unacc":
                    unacc[index] += 1

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-width*3/2, vgood, width, label='vgood', color='green')
    rects2 = ax.bar(x-width/2, good, width, label='good', color='blue')
    rects3 = ax.bar(x+width/2, acc, width, label='acc', color='orange')
    rects4 = ax.bar(x+width*3/2, unacc, width, label='unacc', color='red')
    
    ax.set_ylabel("Number of rows")
    plt.xticks(x, atr_labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    
    fig.tight_layout()
    
    plt.show()
