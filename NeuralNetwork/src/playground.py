from data_loader import load_data
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, confusion_matrix
from testing import test
import seaborn as sns


class_names = [i for i in range(10)]
pred = []

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  pred.append(predicted_label)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def show_confusion_matrix(matrix, class_list):
    """Shows plot with confusion matrix """
    ax = sns.heatmap(matrix, annot=True, fmt='d')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(class_list)
    ax.yaxis.set_ticklabels(class_list)
    plt.show()

if __name__ == "__main__":
    testing_data = load_data('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
    testing_labels = [data[1] for data in testing_data]
    mnist = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10, activation='sigmoid'),
        tf.keras.layers.Dense(10)  
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)
    pred = [np.argmax(i) for i in predictions]
    cm = confusion_matrix(testing_labels, pred, labels=class_names)
    print(cm)
    print(precision_score(testing_labels, pred, labels=class_names, average='micro'))
    
    show_confusion_matrix(cm, class_names)
    
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()
