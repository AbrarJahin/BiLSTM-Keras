from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers import Bidirectional
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from library.BiLstm.Attention import Attention
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, fbeta_score, precision_recall_fscore_support
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class BiLstmBinaryClassifier:
    def __init__(self, maxL, vectorLength):
        self.model, self.attention = self.__createLstmModel(maxL, vectorLength)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        self.model.summary()
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=3,verbose=1, restore_best_weights=True)
        self.modelCheckpoint = ModelCheckpoint(os.path.join("./", "model", "bestBiLstmModel.h5"), monitor="val_loss",verbose=1, save_best_only=True)
        self.classTrainWeight = {
		        0: 1.,
		        1: 1.
	        }

    def train(self, train_generator, validation_generator):
        self.trainHistory = self.model.fit_generator(train_generator, validation_data=validation_generator, epochs=10000, callbacks=[self.modelCheckpoint, self.earlyStopping], class_weight=self.classTrainWeight)

    def _plotPartialScatterPlot(self, pltFig, dataPoints, color = 'green', label = 'Line 1', dimention = 2):
        if dimention == 2:
            aX, aY = self._getListFromEmbedding(dataPoints, dimention)
            pltFig.scatter(aX, aY, color = color, label = label)
        else:	#dimention == 3
            aX, aY, aZ = self._getListFromEmbedding(dataPoints, dimention)
            pltFig.scatter(aX, aY, color = color, label = label)

    def drawTrainTestAccuracyCurve(self):
        trainLoss = [x for i,x in enumerate(self.trainHistory.history['loss'])]
        trainAcc = [x for i,x in enumerate(self.trainHistory.history['acc'])]
        valAcc = [x for i,x in enumerate(self.trainHistory.history['val_acc'])]
        valLoss = [x for i,x in enumerate(self.trainHistory.history['val_loss'])]
        epoch = self.trainHistory.epoch
        ###########################################################
        fig, ax = plt.subplots()

        ax.plot(epoch, trainLoss, color = "blue", label = "Train Loss")
        ax.plot(epoch, trainAcc, color = "red", label = "Train Accuracy")
        ax.plot(epoch, valAcc, color = "green", label = "Validation Accuracy")
        ax.plot(epoch, valLoss, color = "yellow", label = "Validation Loss")

        #plt.title('Train Accuracy = ', str(self.accuracy))
        ax.set_title('Train Accuracy = '+ str(self.accuracy))
        fig.suptitle('ROC-AUC='+str(self.rocAuc)+", F1="+str(self.f1Score), fontsize=10)
        ###########Add more info
        #plt.text(.5, 1, 'self.precision', transform=fig.transFigure, horizontalalignment='center')
        #plt.text(.8, 1, 'self.recall', transform=fig.transFigure, horizontalalignment='center')
        #plt.text(.2, 1, 'self.f1Score', transform=fig.transFigure, horizontalalignment='center')
        ########################
        ax.legend(loc = 'upper left')

        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.show()

    def test(self, testPair, threshold = 0.5):
        yTrue, pred = testPair[1] ,self.model.predict(testPair[0]).tolist()
        yPred = [1 if x[0]>=threshold else 0 for x in pred]
        yProb = [x[0] for x in pred]
        self.accuracy = accuracy_score(yTrue, yPred, normalize=True)
        self.confusionMatrix = confusion_matrix(yTrue, yPred, labels=[0,1], normalize=None)
        '''
        0 correct   |   0 wrong
        1 wrong     |   1 correct
        '''
        #############
        self.classificationReport = classification_report(yTrue, yPred, labels=[0,1])
        print(self.classificationReport)
        #############
        fpr, tpr, threshold = roc_curve(yTrue, yProb)
        self.rocAuc = auc(fpr, tpr)  #Area under curve
        print("AUC = ", self.rocAuc)
        #############
        self.fbetaScore = fbeta_score(yTrue, yPred, average='macro', beta=0.5)
        self.f1Score = f1_score(yTrue, yPred, average='macro')
        self.precision, self.recall, self.fscore, self.support = precision_recall_fscore_support(yTrue, yPred, average='macro', beta=0.5)
        return self.accuracy

    def predict(self, embeddings, threshold = 0.5):
        pred = self.model.predict(embeddings).tolist()
        return [1 if x[0]>=threshold else 0 for x in pred]

    def __createLstmModel(self, maxL, vectorLength):
        net_input = Input(shape=(maxL,vectorLength))
        lstm = Bidirectional(LSTM(256, return_sequences=True))(net_input)
        x, attention = Attention()(lstm)
        dense = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=net_input, outputs=dense)
        track = Model(inputs=net_input, outputs=attention)
        #return model, track
        return model, attention
   