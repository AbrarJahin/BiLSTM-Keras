import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def convertToTensor(x, maxLength=5, embeddingSize=384):
    #if isinstance(x, list):
    #    x = np.array(x)
    #ax = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=maxLength, padding='post', truncating='pre', dtype='float64')
    paddedX = pad_sequences(x, maxlen=maxLength, padding='post', truncating='pre', dtype='float64')
    maskingLayer = tf.keras.layers.Masking(mask_value=0.0)
    maskedX = maskingLayer(paddedX)
    maskValues = maskedX._keras_mask
    return maskedX