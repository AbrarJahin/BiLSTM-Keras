from library.LSH.LSH import LSH
from library.BiLstm.Utils import *
import random
import pandas as pd
import time

csPhraseList = getDataListFromFile('./data/csTerms.csv')
nonCsPhraseList = getDataListFromFile('./data/nonCsTerms.csv')

data = list(set(csPhraseList + nonCsPhraseList))
#random.shuffle(data)
data.sort(key=lambda x : x.lower())
embeddingDict = getSentenceEmbedding(data)
#data = data[:400]
lsh = LSH(data, num_recommendations = 200, permutations = 500)
nearestNeighbor = []

start_time, total = time.time(), len(data)
for i, d in enumerate(data):
	predNearestList = lsh.predict(d)
	predNearestList.remove(d)
	embedding = [embeddingDict[x] for x in predNearestList]
	nearestDataList = getKNearestNeighboursForString(predNearestList, embedding, embeddingDict[d], k=10)
	nearestNeighbor.append(", ".join(str(x) for x in nearestDataList))
	print(i, "of", total, "---- Time:", (time.time()-start_time))

#print(nearestNeighbor)
pd.DataFrame({
	'Phrase': data,
	'Neighbours': nearestNeighbor
	}).to_csv('./output/neighbours.csv', index=False, header=True)