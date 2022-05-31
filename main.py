from library.BiLstm.BiLstmBinaryClassifier import BiLstmBinaryClassifier
from library.BiLstm.Utils import *
#from library.BiLstm.Attention import Attention
#import numpy as np
#np.random.seed(2)
#from keras.models import Model
#from keras.layers import Dense, Input, Dropout, LSTM, Activation
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#from keras.initializers import glorot_uniform
#from keras.models import Sequential
#from keras.layers import Bidirectional
#import copy
#from keras import backend as K
#from keras.layers import Flatten,Permute,RepeatVector,Multiply,Lambda
#import tensorflow as tf
#import spacy 
#import pandas as pd
#import numpy as np

#from tensorflow.keras.utils import Sequence
#import math
#import pickle
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#import matplotlib.pyplot as plt

import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

csPhraseList = csPhraseList[:10]
nonCsPhraseList = nonCsPhraseList[:10]

X, y = getEmbeddingXY(csPhraseList, nonCsPhraseList)
EMBEDDING_SIZE = len(X[0][0])
print("Embedding Size -", EMBEDDING_SIZE)
train_generator, test_generator, validation_generator = getSplittedGenerators(X, y)

lstmModel, attention = BiLstmBinaryClassifier(5,EMBEDDING_SIZE)
lstmModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
lstmModel.summary()

class_weight = {
		0: 1.,
		1: 1.
	}

earlyStopping = EarlyStopping(monitor='val_loss', patience=3,verbose=1, restore_best_weights=True)
modelCheckpoint = ModelCheckpoint(os.path.join("./", "model", "bestBiLstmModel.h5"), monitor="val_loss",verbose=1, save_best_only=True)

runHistory = lstmModel.fit_generator(train_generator, validation_data=validation_generator,epochs=10000, callbacks=[modelCheckpoint, earlyStopping], class_weight=class_weight)

print(attention, runHistory.history)