from keras.layers import Flatten,Permute,RepeatVector,Multiply,Lambda
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers import Flatten,Permute,RepeatVector,Multiply,Lambda
from keras import backend as K

class Attention:
    def __call__(self, input, combine=True, return_attention=True):
        # Expects inp to be of size (?, number of words, embedding dimension)

        repeatSize = int(input.shape[-1])  #512

        # Map through 1 Layer MLP
        denseLayer = Dense(repeatSize, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(input) #none, 52, 512

        # Dot with word-level vector
        denseLayer = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(denseLayer) #none, 52, 1
        denseLayer = Flatten()(denseLayer) # x_a is of shape (?,52,1), we flatten it to be (?,52)
        attentionOutput = Activation('softmax')(denseLayer)

        # Clever trick to do elementwise multiplication of alpha_t with the correct h_t:
        # RepeatVector will blow it out to be (?,52, 512)
        # Then, Permute will swap it to (?,52,512) where each row (?,k,120) is a copy of a_t[k]
        # Then, Multiply performs elementwise multiplication to apply the same a_t to each
        # dimension of the respective word vector
        repeatVector = RepeatVector(repeatSize)(attentionOutput) #none, 512, 52
        repeatVector = Permute([2,1])(repeatVector) #none, 52, 512
        output = Multiply()([input,repeatVector]) #none, 52, 512
        if combine:
        # Now we sum over the resulting word representations
            output = Lambda(lambda x : K.sum(x, axis=1), name='expectation_over_words')(output) #none, 512
        return (output, attentionOutput) if return_attention else output