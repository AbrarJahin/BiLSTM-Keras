import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

class LSH:
	def __init__(self, dataList, num_recommendations = 200, permutations = 256):
		self.permutations = permutations				#Number of Permutations
		self.numRecommendations = num_recommendations	#Number of Recommendations to return
		self.data = dataList
		self.forest = self._getForest(self.data)

	def _preprocess(self, text):
		text = re.sub(r'[^\w\s]','', text)
		tokens = text.lower()
		tokens = tokens.split()
		return tokens

	def _getForest(self, data):
		startTime = time.time()
		allMinHash = []
		for i, text in enumerate(data):
			tokens = self._preprocess(text)
			minHash = MinHash(num_perm=self.permutations)
			for word in tokens:
				minHash.update(word.encode('utf8'))
			allMinHash.append(minHash)
			print(i, time.time()-startTime)
		forest = MinHashLSHForest(num_perm=self.permutations)
		for index, minHash in enumerate(allMinHash):
			forest.add(index, minHash)
		forestIndex = forest.index()
		print('It took %s seconds to build forest.' %(time.time()-startTime))
		return forest

	def predict(self, text, topResultCount = None):
		topResultCount = self.numRecommendations if topResultCount is None else topResultCount
		start_time = time.time()
		tokens = self._preprocess(text)
		minHash = MinHash(num_perm=self.permutations)
		for word in tokens:
			minHash.update(word.encode('utf8'))
		indexList = self.forest.query(minHash, topResultCount)
		output = [self.data[i] for i in indexList]
		#print('It took %s seconds to query forest.' %(time.time()-start_time))
		return output