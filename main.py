from library.BiLstm.BiLstmBinaryClassifier import BiLstmBinaryClassifier
from library.BiLstm.Utils import *
import pandas as pd

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

#csPhraseList = csPhraseList[:len(csPhraseList)//2]
#nonCsPhraseList = nonCsPhraseList[:len(nonCsPhraseList)//2]

csPhraseList = csPhraseList[:20]
nonCsPhraseList = nonCsPhraseList[:20]

X, y = getEmbeddingXY(csPhraseList, nonCsPhraseList)

#################################################################################
iterations = 1
accuracy = 0
for _ in range(iterations):
	EMBEDDING_SIZE, MAX_SEQUENCE_LENGTH = len(X[0][0]), 5
	train_generator, testDataPair, validation_generator = getSplittedGenerators(X, y)
	lstmModel = BiLstmBinaryClassifier(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE)
	summary = lstmModel.model.summary()
	lstmModel.train(train_generator, validation_generator)

	accuracy += lstmModel.test(testDataPair)
	predValues = lstmModel.predict(testDataPair[0])
	#lstmModel.drawTrainTestAccuracyCurve()
	print("Embedding Size = ",str(EMBEDDING_SIZE))
	print(lstmModel.confusionMatrix)

	# Try to extract CS Results and non_CS results
	csX = X[:len(csPhraseList)]
	csAttention = lstmModel.attention(csX)
	csPred = lstmModel.predict(csX)
	nonCsX = X[len(csPhraseList):]
	nonCsAttention = lstmModel.attention(nonCsX)
	nonCsPred = lstmModel.predict(nonCsX)
	###################
	csWrong = getWronglyPredicted(csPhraseList, csPred, 1)
	nonCsWrong = getWronglyPredicted(nonCsPhraseList, nonCsPred, 0)

	pd.DataFrame(csWrong, columns=["data"]).to_csv('./output/CsWrong.csv', index=False, header=None)
	pd.DataFrame(nonCsWrong, columns=["data"]).to_csv('./output/NonCsWrong.csv', index=False, header=None)
print("#################################################################################")
print("Accuracy", accuracy/iterations)

while True:
	if input('Please enter your command: ').lower() == 'exit': break