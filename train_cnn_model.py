from cnn_model import build_tflearn_convnet_1
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from preprocess_data import preprocess_cnn_data
from save_data import save_ttv_cnn_data, save_pck
from load_data import load, load_train_test_cnn_data

# from plot_confusion import plot_confusion_matrix
# import matplotlib.pyplot as plt

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

"""
Function to plot a confusion matrix in tflearn, since inherently there is no such function in the library.
"""


def plot_conf(X_val, y_val, MODEL):
    preds = []
    for image in X_val:
        preds.append(MODEL.predict([image])[0])
    _y = np.argmax(preds, axis=1)
    y = np.argmax(y_val, axis=1)
    cnf_matrix = confusion_matrix(y, _y)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix CNN')
    plt.show()


# load model and evaluate and plot confusion matrix
def evaluate():
    print 'building model...'
    MODEL = build_tflearn_convnet_1()
    print 'building model complete.'
    MODEL.load('./cnn_model_save/CCN')
    plot_conf(load('./cnn_data/X_val.pck'), load('./cnn_data/y_val.pck'), MODEL)


"""
The first two function calls can preprocess the data from scratch and save them.
I have already added the saved and preprocessed files, so to fit the model, the files are needed only to be loaded.
I have also saved my model to directly predict the realtime webcam data/video/image file.
"""

# print 'pre-processing data...'
# preprocess_cnn_data()
# print 'pre-processing data complete.'
#
# print 'save train test val data...'
# save_ttv_cnn_data()
# print 'save train test val data complete.'

print 'loading data...'
X_train, X_test, X_val, y_train, y_test, y_val = load_train_test_cnn_data()
print 'loading data complete.'

# train model and save
print '\nbuilding model...'
MODEL = build_tflearn_convnet_1()
print 'building model complete.'
MODEL.fit(X_train, y_train, n_epoch=10,
          shuffle=True,
          validation_set=(X_test, y_test),
          show_metric=True,
          batch_size=100,
          run_id='CCN')
MODEL.save('./cnn_model_save/CCN2')
