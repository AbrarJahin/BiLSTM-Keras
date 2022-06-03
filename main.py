from library.BiLstm.BiLstmBinaryClassifier import BiLstmBinaryClassifier
from library.BiLstm.Utils import *

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

#csPhraseList = csPhraseList[:20]
#nonCsPhraseList = nonCsPhraseList[:20]

X, y = getEmbeddingXY(csPhraseList, nonCsPhraseList)

#################################################################################

EMBEDDING_SIZE, MAX_SEQUENCE_LENGTH = len(X[0][0]), 5
train_generator, testDataPair, validation_generator = getSplittedGenerators(X, y)
lstmModel = BiLstmBinaryClassifier(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE)
lstmModel.train(train_generator, validation_generator)

accuracy = lstmModel.test(testDataPair)
predVAlues = lstmModel.predict(testDataPair[0])
lstmModel.drawTrainTestAccuracyCurve()

print(attention, trainHistory.history)