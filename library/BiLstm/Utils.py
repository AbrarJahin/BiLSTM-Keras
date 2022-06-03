from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from library.BiLstm.TripletGenereator import TripletGenereator

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
bertEmbeddingModel = pickle.load(open("./model/preTrainedBERT.model", 'rb'))

def getDataListFromFile(fileAddress):
	dataFrame = pd.read_csv(fileAddress, encoding= 'unicode_escape')
	return [x for x in list(set(dataFrame['phrase'])) if isinstance(x, str) and len(x)>0]

def getPaddedWordsFromPhrase(phrase, maxSequenceLength = MAX_DEFAULT_SEQUENCE_LENGTH):
	words = phrase.split(" ")
	return words[:maxSequenceLength]+[""]*max(maxSequenceLength-len(words), 0)

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

def getSplittedGenerators(X, y, trainRatio = 0.6, validationRatio=0.20, embeddingSize = DEFAULT_EMBEDDING_SIZE, maxSequenceLength = MAX_DEFAULT_SEQUENCE_LENGTH, batch_size = DEFAULT_BATCH_SIZE):  #Remaining part is validation ratio
	#print(type(X),isinstance(X, list))
	embeddingSize = len(X[0][0])
	(X_train, X_test, X_val), (y_train, y_test, y_val) = getTrainTestSplit(X, y, trainRatio, validationRatio)
	#print(len(y_train),len(y_test), len(y_val))
	train_generator = TripletGenereator(
		X_train,
		y_train,
		batch_size=batch_size,
		max_length=maxSequenceLength,
		embedding_size=embeddingSize
	)
	test_generator = TripletGenereator(
		X_test,
		y_test,
		batch_size=batch_size,
		max_length=maxSequenceLength,
		embedding_size=embeddingSize
	)
	validation_generator = TripletGenereator(
		X_val,
		y_val,
		batch_size=batch_size,
		max_length=maxSequenceLength,
		embedding_size=embeddingSize
	)
	#return train_generator, test_generator, validation_generator
	return train_generator, (X_test, y_test), validation_generator