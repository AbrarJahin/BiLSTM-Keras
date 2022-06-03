import numpy as np
import math
from tensorflow.keras.utils import Sequence

class TripletGenereator(Sequence):
    def __init__(self, x, y, batch_size=32, max_length=5, embedding_size=384):
        if isinstance(x, list):
          x = np.array(x)
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)
        self.validation_accuracy = []

    def __len__(self):
        # Denotes the number of batches per epoch
        return math.ceil(float(len(self.y)) / self.batch_size)

    def __getitem__(self, idx):
        # Generate one batch of data
        idxes = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.zeros((len(idxes), self.max_length, self.embedding_size))
        y_batch = np.zeros((len(idxes),))
        for i, idx in enumerate(idxes):
            x = self.x[idx]
            if x.shape[0] < self.max_length:
                x = np.vstack([x, np.zeros((self.max_length - x.shape[0], self.embedding_size))])
            else:
                x = x[:self.max_length, :]
            x_batch[i] = x
            y_batch[i] = self.y[idx]
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        np.random.shuffle(self.indices)