import ID3_tree
import playground
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev


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
                try:
                    counter[pred_dict[value]] = 1
                except:
                    pass
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


def tp(matrix, index):
    tp = 0
    tp += matrix[index][index]
    return tp


def fp(matrix, index):
    fp = 0
    for i in range(len(matrix)):
        if i != index:
            fp += matrix[i][index]
    return fp
            
            
def fn(matrix, index):
    fn = 0
    for i in range(len(matrix)):
        if i != index:
            fn += matrix[index][i]
    return fn


def tn(matrix, index):
    tn = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            tn += matrix[j][i]
    tn = tn - fp(matrix, index) - fn(matrix, index) - tp(matrix, index)
    return tn


def fall_out(matrix, index):
    return (fp(matrix, index)/(fp(matrix, index)+tn(matrix, index)))


def accuracy(matrix, index):
    return ((tp(matrix, index)+tn(matrix, index))/(tp(matrix, index)+tn(matrix, index)+fp(matrix, index)+fn(matrix, index)))
    

def precision(matrix, index):
    return (tp(matrix, index)/(tp(matrix, index)+fp(matrix, index)))
    
    
def recall(matrix, index):
    if (tp(matrix, index)+fn(matrix, index)) == 0:
        return 0
    return (tp(matrix, index)/(tp(matrix, index)+fn(matrix, index)))
    
    
def f1(matrix, index):
    if (recall(matrix, index)+precision(matrix, index)) == 0:
        return 0
    return ((2*precision(matrix, index) * recall(matrix, index))/(recall(matrix, index)+precision(matrix, index)))


def fall_out_all(matrix):
    fpa = 0
    tna = 0
    for i in range(len(matrix)):
        fpa += fp(matrix, i)
        tna += tn(matrix, i)
    return (fpa/(fpa+tna))


def accuracy_all(matrix):
    tpa = 0
    tna = 0
    fpa = 0
    fna = 0
    for i in range(len(matrix)):
        fpa += fp(matrix, i)
        tna += tn(matrix, i)
        tpa += tp(matrix, i)
        fna += fn(matrix, i)
    return ((tpa+tna)/(tpa+tna+fpa+fna))
    

def precision_all(matrix):
    tpa = 0
    fpa = 0
    for i in range(len(matrix)):
        fpa += fp(matrix, i)
        tpa += tp(matrix, i)
    return (tpa/(tpa+fpa))
    
    
def recall_all(matrix):
    tpa = 0
    fna = 0
    for i in range(len(matrix)):
        tpa += tp(matrix, i)
        fna += fn(matrix, i)
    return (tpa/(tpa+fna))
    
    
def f1_all(matrix):
    return ((2*precision_all(matrix) * recall_all(matrix))/(recall_all(matrix)+precision_all(matrix)))


def show_confusion_matrix(matrix, class_list):
    """Shows plot with confusion matrix """
    ax = sns.heatmap(matrix, annot=True)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)
    plt.show()


def print_scores(matrix, class_list):
    """Prints class scores of predictions in terminal"""
    for index, cl in enumerate(class_list):
        print("Scores for class " + cl)
        print("Accuracy: " + str(accuracy(matrix, index)))
        print("Precision: " + str(precision(matrix, index)))
        print("Recall: " + str(recall(matrix, index)))
        print("Fall-out: " + str(fall_out(matrix, index)))
        print("F1: " + str(f1(matrix, index)) + '\n')


def overall_scores(matrix):
    print("Scores for all predictions")
    print("Accuracy: " + str(accuracy_all(matrix)))
    print("Precision: " + str(precision_all(matrix)))
    print("Recall: " + str(recall_all(matrix)))
    print("Fall-out: " + str(fall_out_all(matrix)))
    print("F1: " + str(f1_all(matrix)) + '\n')


if __name__ == "__main__":
    data = playground.read_from_csv('data/car.data')
    class_list = list(ID3_tree.count_different_values(data, 'class').keys())
    train, test = playground.train_test_data(data, 0.80, True)
    tree = ID3_tree.ID3_tree(train, 'class')
    pred, trues = pred_true_values([tree], test, 'class')
    ID3_tree.print_tree(tree)
    cn_mat = confusion_matrix(pred, trues, labels=class_list)
    print(cn_mat)
    print("Tree scores")
    print_scores(cn_mat, class_list)
    overall_scores(cn_mat)
    show_confusion_matrix(cn_mat, class_list)
    
    # trees, test_data = ID3_tree.ID3_forest(data, 'class', 5)
    # pred, trues = pred_true_values(trees, test, 'class')
    # cn_mat = confusion_matrix(pred, trues, labels=class_list)
    # print("Forest scores")
    # print_scores(cn_mat, class_list)
    # overall_scores(cn_mat)
    # show_confusion_matrix(cn_mat, class_list)
    
    # y = [[], [], [], [], []]
    # x = [i/100 for i in range(40, 100, 5)]
    labels = ['accuracy', 'precision', 'recall', 'fall_out', 'f1_score']
    
    # for percent in range(40, 100, 5):
    #     z = []
    #     t = []
    #     for i in range(25):
    #         train, test = playground.train_test_data(data, percent/100, True)
    #         tree = ID3_tree.ID3_tree(train, 'class')
    #         pred, trues = pred_true_values([tree], test, 'class')
    #         cn_mat = confusion_matrix(pred, trues, labels=class_list)
    #         z.append(accuracy_all(cn_mat))
    #         t.append(precision_all(cn_mat))
    #     y[0].append(sum(z)/len(z))
    #     y[1].append(sum(t)/len(t))
            
    # fig = plt.figure()
    # for i in range(2):
    #     plt.plot(x, y[i])
    # plt.legend(labels)
    # plt.xlabel("Percent of data as train data")
    # plt.show()
       
       
    # y = [[], []]
    # x = [1,3,5,7,10,20]
    # for k in x:
    #     z = []
    #     t = []
    #     for i in range(25):
    #         trees, test_data = ID3_tree.ID3_forest(data, 'class', k)
    #         pred, trues = pred_true_values(trees, test, 'class')
    #         cn_mat = confusion_matrix(pred, trues, labels=class_list)
    #         z.append(accuracy_all(cn_mat))
    #         t.append(precision_all(cn_mat))
    #     y[1].append(sum(t)/len(t))
    #     y[0].append(sum(z)/len(z))
        
    # fig = plt.figure()
    # for i in range(2):
    #     plt.plot(x, y[i])
    # plt.legend(labels)
    # plt.xlabel("Cross validation times")
    # plt.show()
        
    # x = [1,3,5,7,10,20]
    # acc = []
    # prec = []
    # for k in x:
    #     trees, test_data = ID3_tree.ID3_forest(data, 'class', k)
    #     pred, trues = pred_true_values(trees, test, 'class')
    #     cn_mat = confusion_matrix(pred, trues, labels=class_list)
    #     acc.append(accuracy_all(cn_mat))
    #     prec.append(precision_all(cn_mat))
        
    # print("Accuracy\n")
    # print("min: "+ str(min(acc)))
    # print("max: "+ str(max(acc)))
    # print("srednia: "+ str(sum(acc)/len(acc)))
    # print("odchylenie: "+ str(stdev(acc)))
    # print("Precision\n")
    # print("min: "+ str(min(prec)))
    # print("max: "+ str(max(prec)))
    # print("srednia: "+ str(sum(prec)/len(prec)))
    # print("odchylenie: "+ str(stdev(prec)))
    