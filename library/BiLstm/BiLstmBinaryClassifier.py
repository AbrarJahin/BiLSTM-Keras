from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers import Bidirectional
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from library.BiLstm.Attention import Attention
from keras.models import Model

def BiLstmBinaryClassifier(maxL,vectorLength):
    net_input = Input(shape=(maxL,vectorLength))
    lstm = Bidirectional(LSTM(256, return_sequences=True))(net_input)
    x, attention = Attention()(lstm)
    dense = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=net_input, outputs=dense)
    track = Model(inputs=net_input, outputs=attention)
    #return model, track
    return model, attention

#class BiLstmBinaryClassifier:
#  def __init__(self, _maxLength = 40, _singleEmbeddingSize = 384):
#    net_input = Input(shape=(_maxLength,_singleEmbeddingSize))
#    lstm = Bidirectional(LSTM(256, return_sequences=True))(net_input)
#    x, self.attention = Attention()(lstm)
#    dense = Dense(1, activation="sigmoid")(x)
#    self.model = Model(inputs=net_input, outputs=dense)
#    self.track = Model(inputs=net_input, outputs=self.attention)
   
#  def sayHi(self):
#    print('Hello, my name is ' + self.name + ' and I am ' + self.age + ' years old!')
   