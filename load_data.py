import cPickle

"""
All file loading functions.
"""


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


def load_all_data_ml():
    return load('./ml_data/X.pck'), load('./ml_data/y.pck')


def load_all_data_cnn():
    return load('./cnn_data/X.pck'), load('./cnn_data/y.pck')


def load_train_test_ml_data():
    return load('./ml_data/X_train.pck'), load('./ml_data/X_test.pck'), load('./ml_data/X_val.pck'), \
           load('./ml_data/y_train.pck'), load('./ml_data/y_test.pck'), load('./ml_data/y_val.pck')


def load_train_test_cnn_data():
    return load('./cnn_data/X_train.pck'), load('./cnn_data/X_test.pck'), load('./cnn_data/X_val.pck'), \
           load('./cnn_data/y_train.pck'), load('./cnn_data/y_test.pck'), load('./cnn_data/y_val.pck')
