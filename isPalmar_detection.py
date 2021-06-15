import tensorflow as tf
import cv2 as cv
import numpy as np
tf_model = tf.keras.models.load_model('model.h5')


def rescale(img):
    width = int(img.shape[1] / 30)
    height = int(img.shape[0] / 30)
    dim = (width, height)

    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    return img
def isPalmar(img):
    tf_model = tf.keras.models.load_model('model.h5')
    prediction = tf_model.predict(np.array([rescale(img)]))
    if prediction[0][0] > prediction[0][1]:
        return False
    return True


