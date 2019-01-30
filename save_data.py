import cPickle
from load_data import load_all_data_ml, load_all_data_cnn
from tflearn.data_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

"""
All file saving functions.
"""


def save_pck(cleaned_data, file_name):
    file = open(file_name + '.pck', 'wb')
    cPickle.dump(cleaned_data, file)


def save_ttv_ml_data():
    X, y = load_all_data_ml()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    # saving all the preprocessed data
    save_pck(X_train, './ml_data/X_train')
    save_pck(X_test, './ml_data/X_test')
    save_pck(X_val, './ml_data/X_val')
    save_pck(y_train, './ml_data/y_train')
    save_pck(y_test, './ml_data/y_test')
    save_pck(y_val, './ml_data/y_val')


def save_ttv_cnn_data():
    X, y = load_all_data_cnn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    # converting all data to np arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    y_train = np.copy(y_train)
    y_test = np.copy(y_test)
    y_val = np.copy(y_val)

    # reshaping all image data to feed to the network
    X_train = X_train.reshape([-1, 48, 48, 1])
    X_test = X_test.reshape([-1, 48, 48, 1])
    X_val = X_val.reshape([-1, 48, 48, 1])

    # changing the emotion variables to categorical
    y_train = to_categorical(y_train, 7)
    y_test = to_categorical(y_test, 7)
    y_val = to_categorical(y_val, 7)

    # saving all the preprocessed data
    save_pck(X_train, './cnn_data/X_train')
    save_pck(X_test, './cnn_data/X_test')
    save_pck(X_val, './cnn_data/X_val')
    save_pck(y_train, './cnn_data/y_train')
    save_pck(y_test, './cnn_data/y_test')
    save_pck(y_val, './cnn_data/y_val')
