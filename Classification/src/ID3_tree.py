import math
from copy import deepcopy
import playground
import random


def count_value_appearances(data, label, value):
    """How many times attribute has speciffic value

    Args:
        data (list of dict): data to analyze
        label (string): which attribute 
        value (string): what value we are looking for

    Returns:
        int: how many times attribute has this value 
    """
    counter = 0
    for sample in data:
        if sample[label] == value:
            counter += 1
    return counter


def count_different_values(data, feature):
    """Counts how many different feature values data has

    Args:
        data (list): data to analyze
        feature (string): what feature we are looking for

    Returns:
        dict: how many different values feature has
    """
    counter = {}
    for sample in data:
        if sample[feature] in counter.keys():
            counter[sample[feature]] += 1
        else:
            counter[sample[feature]] = 1
    return counter


def data_with_feature_value(data, feature, value):
    """Takes data from data only with specified value of feature

    Args:
        data (list of dicts): data to analyze
        feature (string): attribute we are looking for
        value (string): value we ae looking for

    Returns:
        list of dicts: data with specified attribute value
    """
    new_data = []
    for row in data:
        if row[feature] == value:
            new_data.append(row)
    return new_data


def remove_data_with_feature_value(data, feature, value):
    """Removes rows from data with specified value of feature

    Args:
        data (list): data to cut
        feature (string): feature to analyze
        value (string): value that we cut off
        
    Returns:
        list: data without rows with specified value of feature
    """
    new_data = []
    for sample in data:
        if sample[feature] != value:
            new_data.append(sample)
    return new_data


def total_entropy(train_data, label, class_list):
    """Entropy of whole data

    Args:
        train_data (list of dicts): data to calculate entropy
        label (string): class name that we evaluate
        class_list (list): values of label

    Returns:
        int: entropy
    """
    total_row = len(train_data)
    entropy = 0
    for cl in class_list:
        class_count = count_value_appearances(train_data, label, cl)
        class_proba = class_count/total_row
        if class_proba != 0:
            class_entropy = -(class_proba) * math.log(class_proba, (2))
            entropy += class_entropy
    return entropy


def spec_entropy(data, label, class_list):
    """Entropy of specified data

    Args:
        data (list): data to calculate entropy
        label (string): class name that we evaluate
        class_list (list): values of label

    Returns:
        int: entropy
    """
    rows_with_value = len(data)
    entropy = 0
    for cl in class_list:
        label_count = count_value_appearances(data, label, cl)
        class_entropy = 0
        if label_count != 0:
            class_proba = label_count/rows_with_value
            class_entropy = -(class_proba) * math.log(class_proba, (2))
        entropy += class_entropy
    return entropy


def info_gain(feature, data, label, class_list):
    """Calculates the information of the feature

    Args:
        feature (string): feature to calculate 
        data (list): data to analyze
        label (string): class name that we evaluate
        class_list (list): values of label

    Returns:
        int: information gain
    """
    feature_values = []
    for row in data:
        if row[feature] not in feature_values:
            feature_values.append(row[feature])
    total_row = len(data)
    info = 0
    for value in feature_values:
        value_data = data_with_feature_value(data, feature, value)
        data_count = len(value_data)
        value_entropy = spec_entropy(value_data, label, class_list)
        value_proba = data_count/total_row
        info += value_proba * value_entropy
    return (total_entropy(data, label, class_list) - info)


def most_valuable_feature(data, label, class_list):
    """Searches for the most valuable feature

    Args:
        data (list): data to analyze
        label (string): class name that we evaluate
        class_list (list): values of label

    Returns:
        string: most valuable feature
    """
    features = list(data[0])
    features = features[:-1]
    max_info_feature = ""
    max_info_gain = -1
    for feature in features:
        info = info_gain(feature, data, label, class_list)
        if max_info_gain < info:
            max_info_gain = info
            max_info_feature = feature
    return max_info_feature


def generate_subtree(feature, data, label, class_list):
    """Generates nofe in tree and values as a branch

    Args:
        feature (string): feature that we want to add to tree
        data (list): data we analyze
        label (string): class name that we evaluate
        class_list (list): values of label
        
    Returns:
        dict, list: tree node with branches and updated data
    """
    features_values_count = count_different_values(data, feature)
    tree = {}
    class_count = {}
    for feature_value, count in features_values_count.items():
        feature_value_data = data_with_feature_value(data, feature, feature_value)
        pure_class = False
        for cl in class_list:
            cl_count = count_value_appearances(feature_value_data, label, cl)
            class_count[cl] = cl_count
            if cl_count == count:
                tree[feature_value] = cl
                data = remove_data_with_feature_value(data, feature, feature_value)
                pure_class = True
        if not pure_class:
            max_value = max(class_count.values())
            classes_to_choose = [key for key, value in class_count.items() if value == max_value]
            pred = random.choice(classes_to_choose)
            tree[feature_value] = '?'+str(pred)
    return tree, data


def make_tree(root, last_feat_value, data, label, class_list):
    """Does tree by recursion

    Args:
        root (dict): currently pointed feature
        last_feat_value (string): last value of pointed feature
        data (list): data to generate tree
        label (string): feature that we looking for
        class_list (list): values of label
    """
    if len(data) != 0:
        max_info_feature = most_valuable_feature(data, label, class_list)
        tree, data = generate_subtree(max_info_feature, data, label, class_list)
        next_root = None
        if last_feat_value != None:
            root[last_feat_value] = dict()
            root[last_feat_value][max_info_feature] = tree
            next_root = root[last_feat_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        for node, branch in next_root.items():
            if '?' in branch:
                data = data_with_feature_value(data, max_info_feature, node)
                make_tree(next_root, node, data, label, class_list)
        return


def ID3_tree(data, label):
    """Generates ID3 tree from provided data

    Args:
        data (list): data to build tree
        label (string): feature we are evaluate

    Returns:
        dict: id3 tree
    """
    train_data = deepcopy(data)
    tree = {}
    class_list = count_different_values(data, label).keys()
    make_tree(tree, None, train_data, label, class_list)
    return tree


def ID3_forest(data, label, times):
    datas = playground.cross_validation(data, times)
    trees = []
    test_data = []
    for data in datas:
        train = data[0]
        test_data += data[1]
        trees.append(ID3_tree(train, label))
    return trees, test_data


if __name__ == "__main__":
    data = playground.read_from_csv('data/car.data')
    tree = ID3_tree(data, 'class')
    print(tree)
