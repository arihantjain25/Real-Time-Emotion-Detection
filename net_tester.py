from cnn_model import build_tflearn_convnet_1
from load_data import load
import numpy as np
from sklearn.metrics import classification_report


def cnn(X_val, y_val):
    print 'building model...'
    MODEL = build_tflearn_convnet_1()
    print 'building model complete.'
    MODEL.load('./cnn_model_save/CCN')
    preds = []
    for image in X_val:
        preds.append(MODEL.predict([image])[0])
    _y = np.argmax(preds, axis=1)
    y = np.argmax(y_val, axis=1)
    print 'Classification report for ConvolutionalNeuralNetwork...'
    print(classification_report(y, _y))


def ml(X_val, y_val):
    rfc = load('./ml_model_saves/rfc.pck')
    rfc_predict = rfc.predict(X_val)
    print '\nClassification report for RandomForestClassifier...'
    print(classification_report(y_val, rfc_predict))

    lda = load('./ml_model_saves/lda.pck')
    lda_predict = lda.predict(X_val)
    print '\nClassification report for LinearDiscriminant...'
    print(classification_report(y_val, lda_predict))

    gnb = load('./ml_model_saves/gnb.pck')
    gnb_predict = gnb.predict(X_val)
    print '\nClassification report for NaiveBayes...'
    print(classification_report(y_val, gnb_predict))

    adc = load('./ml_model_saves/adc.pck')
    adc_predict = adc.predict(X_val)
    print '\nClassification report for AdaBoostClassifier...'
    print(classification_report(y_val, adc_predict))

    svc = load('./ml_model_saves/svm.pck')
    svc_predict = svc.predict(X_val)
    print '\nClassification report for SupportVectorMachines...'
    print(classification_report(y_val, svc_predict))


X_val_ml, y_val_ml = load('./ml_data/X_val.pck'), load('./ml_data/y_val.pck')
X_val_cnn, y_val_cnn = load('./cnn_data/X_val.pck'), load('./cnn_data/y_val.pck')

cnn(X_val_cnn, y_val_cnn)
ml(X_val_ml, y_val_ml)
