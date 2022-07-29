from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from library.BiLstm.TripletGenereator import TripletGenereator
import numpy as np
import heapq
############################################
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
############################################

MAX_DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_EMBEDDING_SIZE = 384

bertEmbeddingModel = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#bertEmbeddingModel = SentenceTransformer('all-mpnet-base-v2')
#bertEmbeddingModel = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
#bertEmbeddingModel = SentenceTransformer('all-distilroberta-v1')
#bertEmbeddingModel = SentenceTransformer("nli-distilroberta-base-v2")

#bertEmbeddingModel = pickle.load(open("./model/preTrainedBERT.model", 'rb'))	#Tuned model

def getCosineDistance(embeddingOrText, secondPhrase = 'Computer Science'):
	input = embeddingOrText
	if isinstance(embeddingOrText, str):
		embeddingOrText = bertEmbeddingModel.encode(embeddingOrText)
	elif isinstance(embeddingOrText, list):
		embeddingOrText = np.array(embeddingOrText)
	passageEmbedding = bertEmbeddingModel.encode(secondPhrase)
	cosineDistanceTensor = util.cos_sim(embeddingOrText, passageEmbedding)
	cosineDistance = cosineDistanceTensor.numpy()[0].tolist()[0]
	return 1 - abs(cosineDistance)

def getCosineDistanceList(embeddingList):
	output = []
	for embedding in embeddingList:
		try:
			output.append(getCosineDistance(embedding))
		except:
			output.append("")
	return [x for x in output]

def getKNearestNeighboursFromCorpus(phraseList, embeddingList, k=10):
	output, orderedOutput = {}, []
	for i, phrase in enumerate(phraseList):
		print(i, '/', len(phraseList))
		phraseEmbedding = bertEmbeddingModel.encode(phrase)
		output[phrase], heap = [], []
		for current in phraseList:
			if phrase == current: continue
			currentEmbedding = bertEmbeddingModel.encode(current)
			similirityTensor = util.cos_sim(phraseEmbedding, currentEmbedding)
			similirity = abs(similirityTensor.numpy()[0].tolist()[0])
			heapq.heappush(heap, (-similirity, current))	#As max heap
		for _ in range(k):
			popData = heapq.heappop(heap)
			output[phrase].append(popData[1])
			orderedOutput.append(popData[1])
	return output

def getKNearestNeighboursForString(phraseList, embeddingList, targetEmbedding, k=10):
	orderedNeighbours, heap = [], []
	for i, phraseEmbedding in enumerate(embeddingList):
		similirityTensor = util.cos_sim(phraseEmbedding, targetEmbedding)
		similirity = abs(similirityTensor.numpy()[0].tolist()[0])
		heapq.heappush(heap, (-similirity, phraseList[i]))	#As max heap
	for _ in range(min(k, len(heap))):
		popData = heapq.heappop(heap)
		orderedNeighbours.append(popData[1])
	return orderedNeighbours

def getDataListFromFile(fileAddress):
	dataFrame = pd.read_csv(fileAddress, encoding= 'unicode_escape')
	return [x for x in list(set(dataFrame['phrase'])) if isinstance(x, str) and len(x)>0]

def getPaddedWordsFromPhrase(phrase, maxSequenceLength = MAX_DEFAULT_SEQUENCE_LENGTH):
	words = phrase.split(" ")
	#return [""] * max(maxSequenceLength-len(words), 0) + words[:maxSequenceLength]	#Pre Padding
	#return words[:maxSequenceLength] + [""] * max(maxSequenceLength-len(words), 0)	#Post Padding
	return words[:maxSequenceLength]

def getPhraseEmbedding(phrase):
	embeddingLength = len(bertEmbeddingModel.encode([""])[0].tolist())
	embedding = []
	for word in getPaddedWordsFromPhrase(phrase):
		if word == "":
			#wordEmbedding = bertEmbeddingModel.encode([""])[0].tolist()
			wordEmbedding = [0]*embeddingLength
		else:
			wordEmbedding = bertEmbeddingModel.encode([word])[0].tolist()
		embedding.append(wordEmbedding)
	#2D tensor or array
	# Add Padding
	return embedding

def getEmbedding(phrases):
	if isinstance(phrases, list):
		embeddings = []
		print("Embedding Starts!")
		for phrase in tqdm(phrases):
			embedding = getPhraseEmbedding(phrase)
			embeddings.append(embedding)
		return embeddings #[x.tolist() for x in embeddings]
	else:
		return getPhraseEmbedding(phrases)

def getSentenceEmbedding(phrases):
	if isinstance(phrases, list):
		embeddingDict = {}
		for phrase in tqdm(phrases):
			embedding = bertEmbeddingModel.encode(phrase)
			embeddingDict[phrase] = embedding
		return embeddingDict
	else:
		return bertEmbeddingModel.encode(phrases)

#Create embedding vectors from phrase
def getEmbeddingXY(csPhraseList, nonCsPhraseList):
	X, y = csPhraseList + nonCsPhraseList, [1]*len(csPhraseList) + [0]*len(nonCsPhraseList)
	return getEmbedding(X), y

def getTrainTestSplit(X, y, trainRatio = 0.2, validationRatio=0.25):
	trainRatio = 1-trainRatio
	validationRatio = validationRatio/trainRatio
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=trainRatio, random_state=1)
	X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=validationRatio, random_state=2) # 0.25 x 0.8 = 0.2
	return (X_train, X_test, X_val), (y_train, y_test, y_val)

def getGenerator(X, y, batch_size, maxSequenceLength, embeddingSize):
	return TripletGenereator(
		X,
		y,
		batchSize=batch_size,
		maxLength=maxSequenceLength,
		embeddingSize=embeddingSize
	)

def getSplittedGenerators(X, y, trainRatio = 0.6, validationRatio=0.20, embeddingSize = DEFAULT_EMBEDDING_SIZE, maxSequenceLength = MAX_DEFAULT_SEQUENCE_LENGTH, batch_size = DEFAULT_BATCH_SIZE):  #Remaining part is validation ratio
	#print(type(X),isinstance(X, list))
	embeddingSize = len(X[0][0])
	(X_train, X_test, X_val), (y_train, y_test, y_val) = getTrainTestSplit(X, y, trainRatio, validationRatio)
	#print(len(y_train),len(y_test), len(y_val))
	train_generator = getGenerator(
								X_train,
								y_train,
								batch_size,
								maxSequenceLength,
								embeddingSize
							)
	test_generator = TripletGenereator(
								X_test,
								y_test,
								batchSize=batch_size,
								maxLength=maxSequenceLength,
								embeddingSize=embeddingSize
							)
	validation_generator = getGenerator(
								X_val,
								y_val,
								batch_size,
								maxSequenceLength,
								embeddingSize
							)
	#return train_generator, test_generator, validation_generator
	return train_generator, (X_test, y_test), validation_generator

def getWronglyPredicted(phraseList, predList, toBePredicted = 1, embedding = None):
	wronglyPredicted, filteredEmbedding = [], []
	for i, v in enumerate(predList):
		if v != toBePredicted:
			wronglyPredicted.append(phraseList[i])
			if embedding: filteredEmbedding.append(embedding[i])
	return wronglyPredicted, filteredEmbedding if embedding else wronglyPredicted