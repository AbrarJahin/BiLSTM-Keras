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
	#print("Embedding Size = ",str(EMBEDDING_SIZE))
	lstmModel.getConfusionMatrix()

	# Try to extract CS Results and non_CS results
	csX = X[:len(csPhraseList)]
	csAttentions = lstmModel.attention(csX)
	csPred = lstmModel.predict(csX)
	nonCsX = X[len(csPhraseList):]
	nonCsAttentions = lstmModel.attention(nonCsX)
	nonCsPred = lstmModel.predict(nonCsX)
	###################
	csWrong = getWronglyPredicted(csPhraseList, csPred, 1)
	nonCsWrong = getWronglyPredicted(nonCsPhraseList, nonCsPred, 0)
	###################
	csWrong, csWrongEmbedding = getWronglyPredicted(csPhraseList, csPred, 1, csX)
	nonCsWrong, nonCsWrongEmbedding = getWronglyPredicted(nonCsPhraseList, nonCsPred, 0, nonCsX)
	###################
	# For csWrong
	csWrongProbabilities = lstmModel.predict(csWrongEmbedding, isReturnProbability=True)
	csWrongAttentions = lstmModel.attention(csWrongEmbedding, isConvertibleToStr = True)
	# For nonCsWrong
	nonCsWrongProbabilities = lstmModel.predict(nonCsWrongEmbedding, isReturnProbability=True)
	nonCsWrongAttentions = lstmModel.attention(nonCsWrongEmbedding, isConvertibleToStr = True)
	#pd.DataFrame(csWrong, columns=["data"]).to_csv('./output/CsWrong.csv', index=False, header=None)
	pd.DataFrame({
		'Phrase': csWrong,
		'Probablity': csWrongProbabilities,
		'Attention': csWrongAttentions
		}).to_csv('./output/CsWrong.csv', index=False, header=True)
	pd.DataFrame({
		'Phrase': nonCsWrong,
		'Probablity': nonCsWrongProbabilities,
		'Attention': nonCsWrongAttentions
		}).to_csv('./output/NonCsWrong.csv', index=False, header=True)
print("#################################################################################")
print("Accuracy", accuracy/iterations)

while True:
	if input('Please enter your command: ').lower() == 'exit': break