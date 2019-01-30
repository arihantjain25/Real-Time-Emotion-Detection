from plot_confusion import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load, load_train_test_ml_data
from save_data import save_pck, save_ttv_ml_data
from preprocess_data import preprocess_ml_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

"""
The two function calls commented can preprocess the data from scratch and save them.
I have already added the save files, so to fit the model, only the files are needed to be loaded.
All the models are trained and saved under the ml_model_saves folder.
"""

# print 'pre-processing data...'
# preprocess_ml_data()
# print 'pre-processing data complete.'
#
# print 'save train test val data...'
# save_ttv_ml_data()
# print 'save train test val data complete.'

print 'loading data...'
X_train, X_test, X_val, y_train, y_test, y_val = load_train_test_ml_data()
print 'loading data complete.'


# Classification using RandomForestClassifier.
# This function saves/loads the model and plots a confusion matrix plot.
def random_forests():
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    save_pck(rfc, './ml_model_saves/rfc')
    # rfc = load('./ml_model_saves/rfc.pck')
    rfc_predict = rfc.predict(X_val)
    print(classification_report(y_val, rfc_predict))
    cnf_matrix = (confusion_matrix(y_val, rfc_predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix RF')
    plt.show()


# Classification using LinearDiscriminant.
# This function saves/loads the model and plots a confusion matrix plot.
def linear_disc():
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    save_pck(lda, './ml_model_saves/lda')
    # lda = load('./ml_model_saves/lda.pck')
    lda_predict = lda.predict(X_val)
    print(classification_report(y_val, lda_predict))
    cnf_matrix = (confusion_matrix(y_val, lda_predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix LDA')
    plt.show()


# Classification using NaiveBayes.
# This function saves/loads the model and plots a confusion matrix plot.
def naive_bayes():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    save_pck(gnb, './ml_model_saves/gnb')
    # gnb = load('./ml_model_saves/gnb.pck')
    gnb_predict = gnb.predict(X_val)
    print(classification_report(y_val, gnb_predict))
    cnf_matrix = (confusion_matrix(y_val, gnb_predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix NB')
    plt.show()


# Classification using SupportVectorMachines.
# This function saves/loads the model and plots a confusion matrix plot.
def support_vector():
    svc = svm.SVC(kernel='linear')
    svc.fit(X_train, y_train)
    save_pck(svc, './ml_model_saves/svm')
    # svc = load('./ml_model_saves/svm.pck')
    svc_predict = svc.predict(X_val)
    print(classification_report(y_val, svc_predict))
    cnf_matrix = (confusion_matrix(y_val, svc_predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix SVM')
    plt.show()


# Classification using AdaBoostClassifier.
# This function saves/loads the model and plots a confusion matrix plot.
def ada_boost():
    adc = AdaBoostClassifier()
    adc.fit(X_train, y_train)
    save_pck(adc, './ml_model_saves/adc')
    # adc = load('./ml_model_saves/adc.pck')
    adc_predict = adc.predict(X_val)
    print(classification_report(y_val, adc_predict))
    cnf_matrix = (confusion_matrix(y_val, adc_predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=EMOTIONS, normalize=True,
                          title='Normalized confusion matrix AB')
    plt.show()


random_forests()
naive_bayes()
linear_disc()
support_vector()
ada_boost()
