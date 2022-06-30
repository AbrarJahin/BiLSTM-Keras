import numpy as np
import math
from tensorflow.keras.utils import Sequence

class TripletGenereator(Sequence):
    def convertToTensor(self, x, maxLength=5, embeddingSize=384):
        if isinstance(x, list):
          x = np.array(x)
        return x

    def __init__(self, x, y, batchSize=32, maxLength=5, embeddingSize=384):
        x = self.convertToTensor(x, maxLength, embeddingSize)
        self.x, self.y = x, y
        self.batchSize = batchSize
        self.maxLength = maxLength
        self.embeddingSize = embeddingSize
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)
        self.validationAccuracy = []

    def __len__(self):
        # Denotes the number of batches per epoch
        return math.ceil(float(len(self.y)) / self.batchSize)

    def __getitem__(self, idx):
        # Generate one batch of data
        idxes = self.indices[idx * self.batchSize:(idx + 1) * self.batchSize]
        xBatch = np.zeros((len(idxes), self.maxLength, self.embeddingSize))
        yBatch = np.zeros((len(idxes),))
        for i, idx in enumerate(idxes):
            x = self.x[idx]
            if x.shape[0] < self.maxLength:
                x = np.vstack([x, np.zeros((self.maxLength - x.shape[0], self.embeddingSize))])
            else:
                x = x[:self.maxLength, :]
            xBatch[i] = x
            yBatch[i] = self.y[idx]
        return np.array(xBatch), np.array(yBatch)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        np.random.shuffle(self.indices)