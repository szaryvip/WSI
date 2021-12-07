import ID3_tree
import playground
import json
import random


def predict(tree, sample):
    """Makes prediction of data sample

    Args:
        tree (dict): decision tree
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


def confusion_matrix(tree, data, label):
    """Generates confusion matrix for tree

    Args:
        tree (dict): tree to analyze
        data (list): data to test tree
        label (string): class that we evaluate

    Returns:
        list: list with dicts of correct label and prediction counter
    """
    con_matrix = []
    class_list = ID3_tree.count_different_values(data, label).keys()
    for cl in class_list:
        con_matrix.append({cl: 0})
        
    return con_matrix


def accuracy(tree, test_data, label):
    correct = 0
    wrong = 0
    for row in test_data:
        result = predict(tree, row)
        if result == row[label]:
            correct += 1
        else:
            wrong += 1
    acc = correct/(correct+wrong)
    return acc


if __name__ == "__main__":
    data = playground.read_from_csv('data/car.data')
    train, test = playground.train_test_data(data, 0.80, True)
    tree = ID3_tree.ID3_tree(train, 'class')
    print(json.dumps(tree, sort_keys=False, indent=2))
    print(accuracy(tree, test, 'class'))
