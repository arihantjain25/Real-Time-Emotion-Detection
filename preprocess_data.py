import pandas as pd
import numpy as np
from save_data import save_pck
import cv2

path = './fer2013.csv'

"""
These functions processes and dumps all the preprocessing for machine learning models and the conv nn model.
"""


def preprocess_ml_data():
    data = pd.read_csv(path)
    pixel_list = data['pixels'].tolist()
    emotions = data['emotion'].tolist()
    faces = []
    for pixels in pixel_list:
        face = string_to_nparray(pixels)
        face = face / float(255)
        faces.append(face)
    save_pck(faces, './ml_data/X')
    save_pck(emotions, './ml_data/y')


def preprocess_cnn_data():
    data = pd.read_csv(path)
    pixel_list = data['pixels'].tolist()
    emotions = data['emotion'].tolist()
    faces = []
    for pixels in pixel_list:
        face = string_to_nparray(pixels)
        face = face.reshape(48, 48)
        face = face / float(255)
        faces.append(face)

    # # to flip the image (commenting out since no significant increase in accuracy).
    # for pixels in pixel_list:
    #     face = string_to_nparray(pixels)
    #     face = face.reshape(48, 48)
    #     face = face / float(255)
    #     faces.append(cv2.flip(face, 1))
    # emotions = emotions + emotions

    save_pck(faces, './cnn_data/X')
    save_pck(emotions, './cnn_data/y')


def string_to_nparray(string):
    string_list = string.split(' ')
    pixels_list = []
    for pixel in string_list:
        pixel = int(pixel)
        pixels_list.append(pixel)
    pixels_arr = np.array(pixels_list)
    return pixels_arr

