import ID3_tree
import playground
import json
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def predict(tree, sample):
    """Makes prediction of data sample

    Args:
        trees (list): decision tree list
        sample (dict): data row

    Returns:
        string: predicted_class
    """
    if not isinstance(tree, dict):
        return tree
    root_node = next(iter(tree))
    feature_value = sample[root_node]
    if feature_value in tree[root_node]:
        pred = predict(tree[root_node][feature_value], sample)
        if '?' in pred:
            pred = pred[1:]
        return pred 
    else:
        pred_dict = tree[root_node]
        counter = {}
        for value in list(pred_dict.keys()):
            if pred_dict[value] not in list(counter.keys()):
                counter[pred_dict[value]] = 1
            else:
                counter[pred_dict[value]] += 1
        max_value = max(counter.values())
        classes_to_choose = [key for key, value in counter.items() if value == max_value]
        pred = random.choice(classes_to_choose)
        return pred


def pred_from_forest(trees, sample):
    """Predicts class that was returned most from trees in forest

    Args:
        trees (list): id3 trees
        sample (dict): data to analyze

    Returns:
        string: prediction of class
    """
    predictions = {}
    for tree in trees:
        pred = predict(tree, sample)
        if pred in list(predictions.keys()):
            predictions[pred] += 1
        else:
            predictions[pred] = 1
    max_value = max(predictions.values())
    classes_to_choose = [key for key, value in predictions.items() if value == max_value]
    pred = random.choice(classes_to_choose)
    return pred
         


def conf_matrix(trees, data, label):
    """Generates confusion matrix for tree

    Args:
        trees (list): trees to analyze
        data (list): data to test trees
        label (string): class that we evaluate

    Returns:
        confusion_matrix: list with dicts of correct label and prediction counter
    """
    predictions = []
    real_values = []
    for sample in data:
        predictions.append(pred_from_forest(trees, sample))
        real_values.append(sample[label])
    return confusion_matrix(predictions, real_values)


def accuracy(trees, test_data, label):
    correct = 0
    wrong = 0
    for row in test_data:
        result = pred_from_forest(trees, row)
        if result == row[label]:
            correct += 1
        else:
            wrong += 1
    acc = correct/(correct+wrong)
    return acc


if __name__ == "__main__":
    data = playground.read_from_csv('data/car.data')
    class_list = ID3_tree.count_different_values(data, 'class').keys()
    train, test = playground.train_test_data(data, 0.80, True)
    tree = ID3_tree.ID3_tree(train, 'class')
    print(json.dumps(tree, sort_keys=False, indent=2))
    print(accuracy([tree], test, 'class'))
    cn_mat = conf_matrix([tree], test, 'class')
    ax = sns.heatmap(cn_mat, annot=True)

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)

    plt.show()
    trees, test_data = ID3_tree.ID3_forest(data, 'class', 5)
    print(accuracy(trees, test_data, 'class'))
    cn_mat = conf_matrix(trees, test_data, 'class')
    ax = sns.heatmap(cn_mat, annot=True)

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)

    plt.show()
    
    
