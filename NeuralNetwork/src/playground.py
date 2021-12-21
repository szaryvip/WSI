from data_loader import load_data
from neural_network import NeuralNetwork

if __name__ == "__main__":
    training_data = load_data(
        'data/train-images.idx3-ubyte',
        'data/train-labels.idx1-ubyte'
    )
    test_data = load_data(
        'data/t10k-images.idx3-ubyte',
        'data/t10k-labels.idx1-ubyte'
    )
    # print(train[0])

    print('Data Loaded')
    # quit()

    Chad = NeuralNetwork(
        hidden_layers_number=2,
        neurons_in_layer_number=10,
        epochs_number=5
    )

    print(
        Chad.back_propagation(
            training_data[:1000],
            test_data[:20],
            1
        )
    )

    print([i[1] for i in test_data[:20]])
