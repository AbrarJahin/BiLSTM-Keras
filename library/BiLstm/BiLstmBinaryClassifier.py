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
import numpy as np
from library.BiLstm.Preprocessing import convertToTensor

class BiLstmBinaryClassifier:
    def __init__(self, maxL, embeddingLength):
        self._maxLength = maxL
        self._embeddingLength = embeddingLength
        self.model, self.attentionLayerModel = self.__createLstmModel(maxL, embeddingLength)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        #self.model.summary()
        # Stop model on early step if accuricy stopps
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
        # Save best model
        self.modelCheckpoint = ModelCheckpoint(os.path.join("./", "model", "bestBiLstmModel.h5"), monitor="val_loss",verbose=1, save_best_only=True)
        # Training weights for each data types
        self.classTrainWeight = {
		        0: 1.,
		        1: 1.
	        }

    def train(self, trainGenerator, validationGenerator):
        self.trainHistory = self.model.fit(trainGenerator, validation_data=validationGenerator, epochs=10000, callbacks=[self.modelCheckpoint, self.earlyStopping], class_weight=self.classTrainWeight)

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
        ax.set_title('Test Accuracy = '+ str(self.accuracy))
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
        testX = convertToTensor(testPair[0], self._maxLength, self._embeddingLength)
        yTrue, pred = testPair[1] ,self.model.predict(testX).tolist()
        yPred = [1 if x[0]>=threshold else 0 for x in pred]
        yProb = [x[0] for x in pred]
        self.accuracy = accuracy_score(yTrue, yPred, normalize=True)
        self._testSize = len(yTrue)
        self._confusionMatrix = confusion_matrix(yTrue, yPred, labels=[0,1], normalize=None).tolist()
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

    #Return the attention layer values
    def attention(self, embeddings, isConvertibleToStr = True):
        try:
            ifSingleData = not isinstance(embeddings[0][0], list)
            if ifSingleData: embeddings = [embeddings]
            valX = convertToTensor(embeddings, self._maxLength, self._embeddingLength)
            pred = self.attentionLayerModel.predict(valX).tolist()
            #valY = [", ".join(map(str,x)) if isConvertibleToStr else x for x in pred]
            valY = [(', '.join('{:.4f}'.format(y) for y in x)) for x in pred] if isConvertibleToStr else pred
            return valY[0] if ifSingleData else valY
        except Exception as ex:
            print(ex)
            return "Wrong Data"

    #Return the prediction values
    def predict(self, embeddings, threshold = 0.5, isReturnProbability = False):
        try:
            ifSingleData = not isinstance(embeddings[0][0], list)
            if ifSingleData: embeddings = [embeddings]
            valX = convertToTensor(embeddings, self._maxLength, self._embeddingLength)
            pred = self.model.predict(valX).tolist()
            valY = [x[0] if isReturnProbability else (1 if x[0]>=threshold else 0) for x in pred]
            return valY[0] if ifSingleData else valY
        except Exception as ex:
            print(ex)
            return 'Data Not Valid'

    def __createLstmModel(self, maxL, vectorLength):
        netInput = Input(shape=(maxL, vectorLength))

        lstmLayer = Bidirectional(LSTM(vectorLength, return_sequences=True))(netInput)  #256 = vectorLength
        attentionLayerOut, attentionLayer = Attention()(lstmLayer)
        dense = Dense(1, activation="sigmoid")(attentionLayerOut)
        model = Model(inputs=netInput, outputs=dense)
        attentionLayerModel = Model(inputs=netInput, outputs=attentionLayer)
        return model, attentionLayerModel

    def getConfusionMatrix(self):
        try:
            '''
            0 correct   |   0 wrong
            1 wrong     |   1 correct
            '''
            #print(self.confusionMatrix)
            print("------------------------------------------------")
            print("Embedding Size = ", self._embeddingLength)
            print("Test Data Size = ", self._testSize)
            print("Non-CS correct = ", self._confusionMatrix[0][0])
            print("Non-CS wrong = ", self._confusionMatrix[0][1])
            print("CS wrong = ", self._confusionMatrix[1][0])
            print("CS correct = ", self._confusionMatrix[1][1])
            print("------------------------------------------------")
        except Exception as ex:
            print("Model test is not run yet!")
            print(ex)