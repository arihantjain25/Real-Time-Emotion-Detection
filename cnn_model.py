import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn import batch_normalization

"""
Input data shape = [None x 48 x 48 x 1]
Used 3 convolutional layers, 2 fully connected layers, 2 max_pools and a dropout.
Used standard L2 regularizers and relu activation functions.
Used dropout and batch normalizations to counter overfitting.
Used adam as the optimizer as it outperformed SGD.
Used 3 as the filter size.
The dropout is half the nodes.
The convlayers incoming tensor is 32, 64 and 64.
Kernel size for the max pool layers is 2.
Fully connected layer nodes are 256 and finally 7 output nodes.
"""


def build_tflearn_convnet_1():
    network = input_data(shape=[None, 48, 48, 1])
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = batch_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.75)
    network = fully_connected(network, 7, activation='softmax')
    network = regression(network,
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return tflearn.DNN(network, tensorboard_verbose=2)
