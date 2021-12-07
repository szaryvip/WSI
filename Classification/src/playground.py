# W ramach tego zadania będą musieli Państwo zaimplementować drzewo decyzyjne przy pomocy algorytmu ID3 
# i przeprowadzić klasyfikację metodą k-krotnej walidacji krzyżowej dla zadanego zbioru danych.  

# Sprawdzić na wykresie czy można na podstawie parametrów obserwacji stwierdzić istnienie poszukiwanych grup klas 
# (może są one liniowo separowalne w jakiejś kombinacji). Sugeruję zawrzeć w raporcie tylko te ciekawsze wykresy (2-3 z komentarzem!).  

# Sprawdzić jak podział danych na zbiór trenujący i testowy wpływa na sprawność modelu (poeksperymentować z proporcjami).
# Czy widoczne są oznaki niedouczenia / przeuczenia modelu? 
# (jeżeli ktoś będzie chciał robić to iteracyjnie nie ma sensu sprawdzić co 0,5-1%, proszę potestować co 5-10% zmian, 
#  aby były widoczne jakieś różnice)  

# Sprawdzić wpływ parametru k na jakość klasyfikacji 
# w przypadku walidacji krzyżowej (k=1,3,5,7,10,20)  

# Porównać efektywność modelu, kiedy zbiór trenujący jest wstępnie 
# posortowany z przypadkiem, gdy dane zostaną specjalnie pomieszane  

# Wyznaczyć macierz błędów (TP, FP, TN, FN) wraz z takimi miarami 
# jak precyzja, czułość i dokładność modelu (sugerowałbym policzyć 
# to ręcznie, ale pozwalam użyć do tego dedykowanych bibliotek)  

# Zamieścić wygląd przykładowego drzewa utworzonego przy pomocy Państwa algorytmu  

 
 

# Państwa implementacja klasyfikatora nie powinna zależeć od zadanego zbioru danych.
# Powinna ona być na tyle ogólna jak się da, aczkolwiek nie wymagam implementacji 
# dodatkowej funkcjonalności, poza tym co jest wymagane w tym zadaniu 
# (tzn. nie muszą Państwo robić wsparcia dla parametrów funkcji, których nie będą używać)  

 
# Zadanie klasyfikacji oceny zakupu samochodu na podstawie informacji 
# o jego specyfikacji. Zbiór tworzy 1728 obserwacji – 
# 4 klasy z czego 1 jest dominująca (1210 obserwacji).  
import ID3_tree
import csv
import random


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


def cross_validation(train_data, times):
    """Divide train data to train and validation data

    Args:
        train_data (dict): data to divide
        times (int): one portion from times portions will be validate data

    Returns:
        list: list of train and validation datas
    """
    border = len(train_data)//times
    data = []
    for i in range(times-1):
        validation_data = train_data[i*border:(i+1)*border]
        train_data = train_data[:i*border] + train_data[(i+1)*border:]
        data.append([train_data, validation_data])
    return data


if __name__ == "__main__":
    pass
