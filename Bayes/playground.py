import matplotlib.pyplot as plt
import pandas as pd
from bayes_classifier import bayes, read_data_from_txt, data_train_test
from sklearn.metrics import confusion_matrix, accuracy_score,\
    precision_score, recall_score, f1_score
import seaborn as sns

class_dict = {1: "Kama", 2: "Rosa", 3: "Canadian"}


def show_heatmap(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='.2g')
    plt.show()


def analyze_dataset(data, arg1: int, arg2: int):
    ax = sns.scatterplot(arg1, arg2, hue=7, data=data)
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['Kama', 'Rosa', 'Canadian'],
              title='Class')
    plt.xlabel(f"Argument {arg1}")
    plt.ylabel(f"Argument {arg2}")
    plt.savefig(path=f"plots/arg{arg1}_{arg2}.png", fname=f"arg{arg1}_{arg2}.png")
    plt.close()


if __name__ == "__main__":
    data = read_data_from_txt("data/seeds_dataset.txt")
    # data = pd.DataFrame(data)
    # for x in range(1, 7, 1):
    #     for y in range(7):
    #         analyze_dataset(data, x, y)
