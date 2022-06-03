from library.BiLstm.BiLstmBinaryClassifier import BiLstmBinaryClassifier
from library.BiLstm.Utils import *

import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

csPhraseList = csPhraseList[:10]
nonCsPhraseList = nonCsPhraseList[:10]

X, y = getEmbeddingXY(csPhraseList, nonCsPhraseList)

train_generator, test_generator, validation_generator = getSplittedGenerators(X, y)

lstmModel, attention = BiLstmBinaryClassifier(5,EMBEDDING_SIZE)
lstmModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
lstmModel.summary()

classTrainWeight = {
		0: 1.,
		1: 1.
	}

earlyStopping = EarlyStopping(monitor='val_loss', patience=3,verbose=1, restore_best_weights=True)
modelCheckpoint = ModelCheckpoint(os.path.join("./", "model", "bestBiLstmModel.h5"), monitor="val_loss",verbose=1, save_best_only=True)

runHistory = lstmModel.fit_generator(train_generator, validation_data=validation_generator, epochs=10000, callbacks=[modelCheckpoint, earlyStopping], class_weight=classTrainWeight)

#lstmModel.predict_generator(test_generator)
pred = lstmModel.predict(test_generator[0])

print(attention, runHistory.history)