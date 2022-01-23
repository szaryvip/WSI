import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bayes_classifier import bayes, read_data_from_txt, data_train_test,\
    correct_classes
from sklearn.metrics import confusion_matrix, accuracy_score,\
    precision_score, recall_score, f1_score
import seaborn as sns

class_dict = {1: "Kama", 2: "Rosa", 3: "Canadian"}
labels = ["Kama", "Rosa", "Canadian"]


def show_heatmap(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='.2g',
                xticklabels=labels, yticklabels=labels)
    plt.show()


def analyze_dataset(data, arg1: int, arg2: int):
    ax = sns.scatterplot(arg1, arg2, hue=7, data=data)
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['Kama', 'Rosa', 'Canadian'],
              title='Class')
    plt.xlabel(f"Argument {arg1}")
    plt.ylabel(f"Argument {arg2}")
    plt.savefig(path=f"plots/arg{arg1}_{arg2}.png",
                fname=f"arg{arg1}_{arg2}.png")
    plt.close()


def make_plots_with_atributes():
    data = read_data_from_txt("data/seeds_dataset.txt")
    data = pd.DataFrame(data)
    for x in range(7):
        for y in range(7):
            if (x != y):
                analyze_dataset(data, x, y)


def data_separation_analyze():
    x = np.arange(0.2, 0.96, 0.05)
    acc = []
    prec = []
    for border in x:
        acc_to_mean = []
        prec_to_mean = []
        for _ in range(25):
            data = read_data_from_txt("data/seeds_dataset.txt")
            train, test = data_train_test(data, border)
            correct = correct_classes(test)
            predictions = bayes(train, test)
            a = accuracy_score(correct, predictions)
            p = precision_score(correct, predictions, average="macro")
            acc_to_mean.append(a)
            prec_to_mean.append(p)
        acc.append(np.mean(acc_to_mean))
        prec.append(np.mean(prec_to_mean))
    plt.plot(x, acc, label="accuracy")
    plt.plot(x, prec, label="precision")
    plt.legend()
    plt.xlabel("Percent of train data")
    plt.ylabel("Metrics score")
    plt.show()


def shuffle_analyze():
    acc_sh = []
    prec_sh = []
    f1_sh = []
    recall_sh = []
    for _ in range(25):
        data = read_data_from_txt("data/seeds_dataset.txt")
        train, test = data_train_test(data, 0.85)
        correct = correct_classes(test)
        predictions = bayes(train, test)
        a = accuracy_score(correct, predictions)
        p = precision_score(correct, predictions, average="macro")
        f1 = f1_score(correct, predictions, average="macro")
        r = recall_score(correct, predictions, average="macro")
        acc_sh.append(a)
        prec_sh.append(p)
        f1_sh.append(f1)
        recall_sh.append(r)
    print("==========Shuffled==========\n")
    print("Acc: ", np.mean(acc_sh), "\n")
    print("Prec: ", np.mean(prec_sh), "\n")
    print("F1: ", np.mean(f1_sh), "\n")
    print("Recall: ", np.mean(recall_sh), "\n")
    acc_not = []
    prec_not = []
    f1_not = []
    recall_not = []
    for _ in range(25):
        data = read_data_from_txt("data/seeds_dataset.txt", False)
        train, test = data_train_test(data, 0.85)
        correct = correct_classes(test)
        predictions = bayes(train, test)
        a = accuracy_score(correct, predictions)
        p = precision_score(correct, predictions, average="macro")
        f1 = f1_score(correct, predictions, average="macro")
        r = recall_score(correct, predictions, average="macro")
        acc_not.append(a)
        prec_not.append(p)
        f1_not.append(f1)
        recall_not.append(r)
    print("=======Not Shuffled=========\n")
    print("Acc: ", np.mean(acc_not), "\n")
    print("Prec: ", np.mean(prec_not), "\n")
    print("F1: ", np.mean(f1_not), "\n")
    print("Recall: ", np.mean(recall_not), "\n")


def efficiency():
    data = read_data_from_txt("data/seeds_dataset.txt")
    train, test = data_train_test(data, 0.85)
    correct = correct_classes(test)
    predictions = bayes(train, test)
    acc = accuracy_score(correct, predictions)
    prec = precision_score(correct, predictions, average="macro")
    f1 = f1_score(correct, predictions, average="macro")
    recall = recall_score(correct, predictions, average="macro")
    print("Acc: ", acc, "\n")
    print("Prec: ", prec, "\n")
    print("F1: ", f1, "\n")
    print("Recall: ", recall, "\n")
    con_mat = confusion_matrix(correct, predictions)
    show_heatmap(con_mat)


if __name__ == "__main__":
    efficiency()
