from library.BiLstm.BiLstmBinaryClassifier import BiLstmBinaryClassifier
from library.BiLstm.Utils import *
import pandas as pd
from library.BiLstm.WikipediaDownload import getArticleList

#distance = getCosineDistance("map function")

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

#csPhraseList = csPhraseList[:len(csPhraseList)//200]
#nonCsPhraseList = nonCsPhraseList[:len(nonCsPhraseList)//200]

#csPhraseList = csPhraseList[:20]
#nonCsPhraseList = nonCsPhraseList[:20]

totalPhrases = csPhraseList + nonCsPhraseList

X, y = getEmbeddingXY(csPhraseList, nonCsPhraseList)

#wikiDownload = getArticleList(totalPhrases)
#kNearestCorpusDict = {}
## Comment this line if need to run faster
#kNearestCorpusDict = getKNearestNeighboursFromCorpus(totalPhrases, X)

#################################################################################
iterations = 3
accuracy = 0
for iteration in range(iterations):
	EMBEDDING_SIZE, MAX_SEQUENCE_LENGTH = len(X[0][0]), 5
	train_generator, testDataPair, validation_generator = getSplittedGenerators(X, y)
	lstmModel = BiLstmBinaryClassifier(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE)
	summary = lstmModel.model.summary()
	lstmModel.train(train_generator, validation_generator)

	accuracy += lstmModel.test(testDataPair)
	if iteration==iterations-1:		#Last Iteration
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
		####
		csWrong = getWronglyPredicted(csPhraseList, csPred, 1)
		nonCsWrong = getWronglyPredicted(nonCsPhraseList, nonCsPred, 0)
		####
		csWrong, csWrongEmbedding = getWronglyPredicted(csPhraseList, csPred, 1, csX)
		nonCsWrong, nonCsWrongEmbedding = getWronglyPredicted(nonCsPhraseList, nonCsPred, 0, nonCsX)
		##############################################################################
		# For csWrong
		csWrongProbabilities = lstmModel.predict(csWrongEmbedding, isReturnProbability=True)
		csWrongAttentions = lstmModel.attention(csWrongEmbedding, isConvertibleToStr = True)
		pd.DataFrame({
			'IfCS': [1]*len(csWrong),
			'Phrase': csWrong,
			'Correct': ['']*len(csWrong),
			'Probablity': csWrongProbabilities,
			'Attention': csWrongAttentions,
			'Distance': getCosineDistanceList(csWrong),
			#'NearestInCorpus': [', '.join(str(e) for e in kNearestCorpusDict[x]) for x in csWrong] if x in kNearestCorpusDict else ['']*len(csWrong)
			}).to_csv('./output/CsWrong.csv', index=False, header=True)
		##############################################################################
		# For nonCsWrong
		nonCsWrongProbabilities = lstmModel.predict(nonCsWrongEmbedding, isReturnProbability=True)
		nonCsWrongAttentions = lstmModel.attention(nonCsWrongEmbedding, isConvertibleToStr = True)
		#pd.DataFrame(csWrong, columns=["data"]).to_csv('./output/CsWrong.csv', index=False, header=None)
		pd.DataFrame({
			'IfCS': [0]*len(nonCsWrong),
			'Phrase': nonCsWrong,
			'Correct': ['']*len(nonCsWrong),
			'Probablity': nonCsWrongProbabilities,
			'Attention': nonCsWrongAttentions,
			'Distance': getCosineDistanceList(nonCsWrong),
			#'NearestInCorpus': [', '.join(str(e) for e in kNearestCorpusDict[x]) for x in nonCsWrong] if x in kNearestCorpusDict else ['']*len(nonCsWrong)
			}).to_csv('./output/NonCsWrong.csv', index=False, header=True)
		##############################################################################
		#CS-CS-> CS Correct
		csCorrect = getWronglyPredicted(csPhraseList, csPred, 0)
		csCorrect, csCorrectEmbedding = getWronglyPredicted(csPhraseList, csPred, 0, csX)
		csCorrectProbabilities = lstmModel.predict(csCorrectEmbedding, isReturnProbability=True)
		csCorrecctAttentions = lstmModel.attention(csCorrectEmbedding, isConvertibleToStr = True)
		pd.DataFrame({
			'IfCS': [1]*len(csCorrect),
			'Phrase': csCorrect,
			'Correct': ['']*len(csCorrect),
			'Probablity': csCorrectProbabilities,
			'Attention': csCorrecctAttentions,
			'Distance': getCosineDistanceList(csCorrect),
			#'NearestInCorpus': [', '.join(str(e) for e in kNearestCorpusDict[x]) for x in csCorrect] if x in kNearestCorpusDict else ['']*len(csCorrect)
			}).to_csv('./output/CsCorrect.csv', index=False, header=True)
		##############################################################################
		#non-CS-NocCS -> NonCS Correct
		nonCsCorrect = getWronglyPredicted(nonCsPhraseList, nonCsPred, 1)
		nonCsCorrect, nonCsCorrectEmbedding = getWronglyPredicted(nonCsPhraseList, nonCsPred, 1, nonCsX)
		nonCsCorrectProbabilities = lstmModel.predict(nonCsCorrectEmbedding, isReturnProbability=True)
		nonCsCorrecctAttentions = lstmModel.attention(nonCsCorrectEmbedding, isConvertibleToStr = True)
		pd.DataFrame({
			'IfCS': [0]*len(nonCsCorrect),
			'Phrase': nonCsCorrect,
			'Correct': ['']*len(nonCsCorrect),
			'Probablity': nonCsCorrectProbabilities,
			'Attention': nonCsCorrecctAttentions,
			'Distance': getCosineDistanceList(nonCsCorrect),
			#'NearestInCorpus': [', '.join(str(e) for e in kNearestCorpusDict[x]) for x in nonCsCorrect] if x in kNearestCorpusDict else ['']*len(nonCsCorrect)
			}).to_csv('./output/NonCsCorrect.csv', index=False, header=True)
print("#################################################################################")
print("Accuracy", accuracy/iterations)

while True:
	if input('Please enter your command: ').lower() == 'exit': break