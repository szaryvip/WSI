# Recall  TPR = TP / (TP + FN) - ile ze wszystkich naprawdę pozytywnych 
# przypadków udało nam się znaleźć (tzn. zaklasyfikować jako pozytywne). 
# Ważne, jeśli zależy nam na wykrywalności a nie interesują nas fałszywe alarmy. 
# Wadą jest to, że ignoruje połowę przypadków, więc model, który zawsze będzie 
# zwracał true będzie miał TPR=1, co jest bezwartościowe.

# Fall-out FPR = FP / (FP + TN) - stosunek fałszywych alarmów 
# (ile niepoprawnie zaklasyfikowano pozytywnych ze wszystkich negatywnych).  

# Precision PPV = TP / (TP + FP) - ile z przewidzianych pozytywnych przypadków 
# naprawdę jest pozytywnych. Wadą tej miary będzie przypadek, gdy dla zadania klasyfikacji 
# "czy ktoś jest nieletni" to wszystkich emerytów zaklasyfikuje jako pełnoletnich a resztę 
# obserwacji zignoruje. Recall będzie mały, ale Precision maksymalne. 

# Accuracy ACC = (TP + TN) / (TP + TN + FP + FN) - ilość poprawnie 
# przewidzianych w stosunku do wszystkich predykcji. 
# Przykład kiedy to się nie sprawdza - dla lotów samolotowych w USA 
# oznaczmy każdego pasażera jako nie-terrorysta. 
# Przyjmując, że w latach 2000-2017 było średnio 800 mln pasażerów 
# i tylko 17 wykrytych terrorystów to nasz model potrafi
# ich identyfikować z ACC = 99,99999% 

# F1-score  F1 = (2 * Precision * Recall) / (Recall + Precision) -
# próba stworzenia miary w oparciu o Recall 
# i Precision (średnia harmoniczna). Pozwala porównywać modele, 
# gdy jeden z tych parametrów jest mały a drugi duży. 
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
    labels = ["vgood", "good", "acc", "unacc"]
    atr_labels = ["low", "med", "high"]
    colors = ["green", "blue", "yellow", "red"]
    x = [[0, 0, 0],[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for row in data:
        for index,atr in enumerate(atr_labels):
            if row["safety"] == atr:
                if row["class"] == "vgood":
                    x[0][index] += 1
                if row["class"] == "good":
                    x[1][index] += 1
                if row["class"] == "acc":
                    x[2][index] += 1
                if row["class"] == "unacc":
                    x[3][index] += 1
                
    fig = plt.figure()
    x_multi = [np.random.randn(n) for n in [10000, 5000, 2000, 100]]
    print(x_multi)
    plt.hist(x_multi, 3, histtype="bar", label=labels, color=colors)
    plt.xlabel("safety")
    plt.ylabel("number of cars")
    plt.legend()
    plt.show()
