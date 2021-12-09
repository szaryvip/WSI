import ID3_tree
import playground
import random
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
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
         

def pred_true_values(trees, data, label):
    """Generate two list. First with predicted values,
    second with real values from data.      

    Args:
        trees (list): trees for predictions, if only one use []
        data (list): data to predict
        label (string): class to predict

    Returns:
        list, list: list of predictions and list of real values
    """
    predictions = []
    real_values = []
    for sample in data:
        predictions.append(pred_from_forest(trees, sample))
        real_values.append(sample[label])
    return predictions, real_values


def tp(matrix):
    tp = 0
    for i in range(len(matrix)):
        tp += matrix[i][i]
    return tp


def fp(matrix):
    fp = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                fp += matrix[i][j]
    return fp
            
            
def fn(matrix):
    fn = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                fn += matrix[j][i]
    return fn


def tn(matrix):
    tn = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            tn += matrix[j][i]
    tn *= len(matrix)
    tn = tn - fp(matrix) - fn(matrix) - tp(matrix)
    return tn


def fall_out(matrix):
    return (fp(matrix)/(fp(matrix)+tn(matrix)))


def accuracy(matrix):
    return ((tp(matrix)+tn(matrix))/(tp(matrix)+tn(matrix)+fp(matrix)+fn(matrix)))
    

def precision(matrix):
    return (tp(matrix)/(tp(matrix)+fp(matrix)))
    
    
def recall(matrix):
    return (tp(matrix)/(tp(matrix)+fn(matrix)))
    
    
def f1(matrix):
    return ((2*precision(matrix) * recall(matrix))/(recall(matrix)+precision(matrix)))


def show_confusion_matrix(matrix, class_list):
    """Shows plot with confusion matrix """
    ax = sns.heatmap(matrix, annot=True)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)
    plt.show()


def print_scores(matrix):
    """Prints scores of predictions in terminal"""
    print("Accuracy: " + str(accuracy(matrix)))
    print("Precision: " + str(precision(matrix)))
    print("Recall: " + str(recall(matrix)))
    print("Fall-out: " + str(fall_out(matrix)))
    print("F1: " + str(f1(matrix)) + '\n')


if __name__ == "__main__":
    data = playground.read_from_csv('data/car.data')
    class_list = list(ID3_tree.count_different_values(data, 'class').keys())
    train, test = playground.train_test_data(data, 0.80, True)
    
    tree = ID3_tree.ID3_tree(train, 'class')
    pred, trues = pred_true_values([tree], test, 'class')
    # ID3_tree.print_tree(tree)
    cn_mat = confusion_matrix(pred, trues)
    print("Tree scores")
    print_scores(cn_mat)
    
    show_confusion_matrix(cn_mat, class_list)
    
    trees, test_data = ID3_tree.ID3_forest(data, 'class', 5)
    pred, trues = pred_true_values(trees, test, 'class')
    cn_mat = confusion_matrix(pred, trues)
    print("Forest scores")
    print_scores(cn_mat)
    
    show_confusion_matrix(cn_mat, class_list)
    
    
