from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from library.BiLstm.TripletGenereator import TripletGenereator

maxLength = 40
batch_size = 128
EMBEDDING_SIZE = 384

bertEmbeddingModel = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def getDataListFromFile(fileAddress):
	dataFrame = pd.read_csv(fileAddress, encoding= 'unicode_escape')
	return [x for x in list(set(dataFrame['phrase'])) if isinstance(x, str) and len(x)>0]

def getPaddedWordsFromPhrase(phrase, length = 5):
	words = phrase.split(" ")
	return words[:5]+[""]*(5-len(words))

def getPhraseEmbedding(phrase):
	embedding = []
	for word in getPaddedWordsFromPhrase(phrase):
		if word == "":
			wordEmbedding = [0]*384   #length of embedding is 384
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

def getSplittedGenerators(X, y, batch_size=128, trainRatio = 0.6, validationRatio=0.17):  #Remaining part is validation ratio
	(X_train, X_test, X_val), (y_train, y_test, y_val) = getTrainTestSplit(X, y, trainRatio, validationRatio)
	print(len(y_train),len(y_test), len(y_val))
	train_generator = TripletGenereator(
		X_train,
		y_train,
		batch_size=batch_size,
		max_length=maxLength,
		embedding_size=EMBEDDING_SIZE
	)
	test_generator = TripletGenereator(
	X_test,
	y_test,
	batch_size=batch_size,
	max_length=maxLength,
	embedding_size=EMBEDDING_SIZE
	)
	validation_generator = TripletGenereator(
		X_val,
		y_val,
		batch_size=batch_size,
		max_length=maxLength,
		embedding_size=EMBEDDING_SIZE
	)
	return train_generator, test_generator, validation_generator
