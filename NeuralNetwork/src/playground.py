# Należałoby uzyskane rezultaty porównać dla różnej liczby neuronów ukrytych 
# i różnej liczby warstw (sprawdzić tylko kilka przypadków z uwagi na czas nauki sieci). 
# Warto sprawdzić też wpływ współczynnika uczenia i ilości epok na pracę modelu. 
# W raporcie należy opisać wygląd przykładowej sieci i przeprowadzić wstępną analizę 
# zbioru danych. Pokazać jakie przykłady ze zbioru MNIST zostały sklasyfikowane
# poprawnie a jakie nie (dosłownie kilka obrazów np. że cyfry 1 i 7 są błędne 
# albo że 2 jest wykrywane poprawnie). Oczywiście nie należy zapomnieć o macierzy 
# pomyłek i miarach jakości klasyfikacji.

# Uwagi do pracy:
# - wykonują Państwo to zadanie w parach dlatego w raporcie należy 
# wykazać kto wykonał jakie zadania
# - rozwiązanie przesyła do mnie tylko jedna osoba z zespołu
# - podobnie jak w poprzednich ćwiczeniach nie chcę otrzymywać
# od Państwa zbioru danych
# - na potrzebę propagacji wstecznej przyda się Państwu przypomnieć 
# wiadomości z algorytmu spadku gradientu z pierwszych ćwiczeń
# - sugerowałbym użyć bibliotekę numpy do reprezentacji danych 
# (bardziej dla Państwa wygody, to nie jest wymóg jako taki) 

# Bonusowo mogą Państwo w ramach ciekawości porównać wyniki 
# uzyskane w ramach implementacji z pytorch albo keras/tensorflow 
# i opisać w raporcie (brak tego nie będzie negatywnie oceniany, 
# natomiast może to pozytywnie wpłynąć na ocenę, jeśli będą mieli 
# Państwo tylko drobne błędy w pracy. Podstawą oceny za to ćwiczenie
# jest bazowa implementacja sieci neuronowej) 

# Podpowiedź do zadania: 

# W skład sieci neuronowej wchodzi:

# warstwa wejściowa (w naszym przypadku 784 piksele) 
# neurony warstwy ukrytej 
# wagi i wartość progu między warstwami 
# funkcja aktywacji (możemy założyć funkcję sigmoidalną) 
# warstwa wyjściowa (w naszym przypadku docelowa klasa czyli 0-9 - warto użyć softmax) 
 
# Model inicjalizowany jest losowymi wagami i w trakcie nauki modelu 
# wykorzystuje się propagację i wsteczną propagację gradientu by je 
# odpowiednio ustawić. W skrócie obliczenia na macierzach i redukcja 
# błędu predykcji. W pewnym momencie dostaniemy sensowne wagi, które będą 
# oddawały naturę naszego problemu. Oczywiście jak będziemy dalej 
# kontynuować ten proces to przeuczymy sieć.  

# Jednym z problemów, który mogą Państwo zaobserwować w swoim r
# ozwiązaniach może być problem zaniku gradientu. Dzieje się to,
# gdy wagi sieci wstępnie ustawimy na zero lub przyrost gradientu
# jest bardzo mały. Stąd też współcześnie zamiast funkcji sigmoidalnej 
# używa się ReLU, tanH lub innych tego typu funkcji. W Państwa przypadku 
# nie ma to aż tak dużego znaczenia, ale warto o tym pamiętać. 

import numpy as np
import idx2numpy
from copy import deepcopy
import matplotlib.pyplot as plt


def normalization(data):
    pass


def load_data_train(path):
    data = idx2numpy.convert_from_file(path)
    data = deepcopy(data)
    # [[wiersz danych], label] --SZARY
    # data[0][0] -- wiersz danych a data[0][1] -- label
    # normalizacja TODO
    np.random.shuffle(data)
    return data


def load_data_test(path):
    pass


if __name__ == "__main__":
    train = load_data_train('data/train-images.idx3-ubyte')

#TODO
# klasa neuralnetwork -- nazwy jak na tej stronce https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/ potem pozmieniac
# -init_network -- Szymon
    # -inicjalizacja wag losowych (wylosować wagi randomowo)
# -train_network 
    # -funkcja aktywacji (wagi * input) -- szymon
    # -transfer (przerzucam to co wyszlo z funkcji aktywacji przez np sigmoida) --szymon
    # -forward_propagate --szymon
    # -transfer_derivate (output do pochodnej) --szary
    # -backward_propagate_error (liczymy blad) --szary
    # -update_weights -- szary
# -backward_propagation --szymon
    # -predict

# czesc testujaca (macierze itp do sprawka) -- Szymon
# pisanie sprawka -- szary

# porownanie z tensorflow -- do ustalenia (dodatkowa)

