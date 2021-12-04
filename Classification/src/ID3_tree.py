import math


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
        if data[label] == value:
            counter += 1
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
        class_entropy = -(class_count/total_row) * math.log(class_count, (2))
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
    features = data[0].keys()
    features = features[:-1]
    max_info_feature = ""
    max_info_gain = -1
    for feature in features:
        info = info_gain(feature, data, label, class_list)
        if max_info_gain < info:
            max_info_gain = info
            max_info_feature = feature
    return max_info_feature


def generate_sub_tree():
    pass


def ID3_tree():
    pass
