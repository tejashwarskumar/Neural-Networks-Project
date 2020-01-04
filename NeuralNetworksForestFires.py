import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

forestData = pd.read_csv("C:/My Files/Excelr/15 - Neural Networks/Assignment/forestfires.csv")
forestData.head()
forestData.describe()

forestData.columns
forestData.drop('month',inplace=True,axis=1)
forestData.drop('day',inplace=True,axis=1)

forestData.isnull().sum() 

forestData['size_category'].value_counts()
forestData['size_category'].value_counts().plot(kind="bar")

#convert size to dummy variable
forestData['size_category'] = np.where(forestData['size_category'] == "small",0,1)

from sklearn.model_selection import train_test_split
train,test = train_test_split(forestData,test_size=0.3)
trainX = train.drop('size_category',axis=1)
trainY = train['size_category']
testX = test.drop('size_category',axis=1)
testY = test['size_category']

#standardize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30))
mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)

from sklearn.metrics import confusion_matrix
confusion_matrix(prediction_train,trainY)
accuracy_train = (263+84)/(263+14+0+84)
accuracy_train

#Test on test Data
prediction_test=mlp.predict(testX)
confusion_matrix(prediction_test,testY)
accuracy_test = (105+7)/(105+34+10+7)
accuracy_test

#try with different activation functions
activation = ['identity', 'logistic', 'tanh', 'relu']
acc = [];
for i in range(0,4):
    print(activation[i])
    model_i = MLPClassifier(hidden_layer_sizes=(30,30),activation = activation[i])
    model_i.fit(trainX,trainY)
    prediction_train_l=model_i.predict(trainX)
    prediction_test_l=mlp.predict(testX)
    accurancy_train_i = np.mean(prediction_train_l == trainY)
    accurancy_test_i = np.mean(prediction_test_l == testY)
    acc.append([accurancy_train_i,accurancy_test_i])

plt.plot(np.arange(0,4),[i[0] for i in acc],"bo-")
plt.plot(np.arange(0,4),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])

#Without standardization
train2,test2 = train_test_split(forestData)
trainX2 = train.drop('size_category',axis=1)
trainY2 = train['size_category']
testX2 = test.drop('size_category',axis=1)
testY2 = test['size_category']

mlp2 = MLPClassifier(hidden_layer_sizes=(30,30))
mlp2.fit(trainX2,trainY2)
prediction_train2=mlp2.predict(trainX2)
accuracy2 = np.mean(prediction_train2 == trainY2)
accuracy2
prediction_test2=mlp2.predict(testX2)
accuracy_test2 = np.mean(prediction_test2 == testY2)
accuracy_test2

activation2 = ['identity', 'logistic', 'tanh', 'relu']
acc2 = [];
for i in range(0,4):
    print(activation[i])
    model_i = MLPClassifier(hidden_layer_sizes=(30,30),activation = activation[i])
    model_i.fit(trainX2,trainY2)
    prediction_train_l=model_i.predict(trainX2)
    prediction_test_l=mlp.predict(testX2)
    accurancy_train_i = np.mean(prediction_train_l == trainY2)
    accurancy_test_i = np.mean(prediction_test_l == testY2)
    acc2.append([accurancy_train_i,accurancy_test_i])

plt.plot(np.arange(0,4),[i[0] for i in acc],"bo-")
plt.plot(np.arange(0,4),[i[1] for i in acc],"ro-")
plt.legend(["train2","test2"])
