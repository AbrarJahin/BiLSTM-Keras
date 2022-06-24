import tensorflow as tf
import numpy as np

class WriteLayerValCallback(tf.keras.callbacks.Callback):
   def __init__(self):
        self.data = np.random.rand(1,10)

   def on_epoch_end(self, epoch, logs=None):
        #dns_layer = self.model.layers[6]
        dns_layer = self.model.get_layer('activation')
        outputs = dns_layer(self.data)
        tf.print(f'\n input: {self.data}')
        tf.print(f'\n output: {outputs}')