import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

startUpData = pd.read_csv("C:/My Files/Excelr/15 - Neural Networks/Assignment/50_Startups.csv")
startUpData.head()
startUpData.describe()

plt.hist(startUpData.Profit)
plt.boxplot(startUpData.Profit)

startUpData.Profit.describe()
startUpData.shape
startUpData = startUpData.drop(startUpData.index[[49]])
plt.boxplot(startUpData.Profit)
startUpData.isnull().sum()

startUpData['State'].value_counts()

state_dummies = pd.get_dummies(startUpData['State'])
startUpData = pd.concat([startUpData,state_dummies],axis=1)

startUpData.drop('State',axis=1, inplace=True)
pd.set_option('display.expand_frame_repr', False)
startUpData.columns
startUpData =  startUpData[['R&D Spend', 'Administration', 'Marketing Spend','California', 'Florida','New York','Profit']]

from sklearn.model_selection import train_test_split
train,test = train_test_split(startUpData,test_size=0.3)
trainX = train.iloc[:,0:6]
trainY = train.iloc[:,6]
testX = test.iloc[:,0:6]
testY = test.iloc[:,6]

from keras.models import Sequential
from keras.layers import Dense
def prep_model(hidden_dim):
    print(hidden_dim)
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

first_model = prep_model([6,50,1])
first_model.fit(trainX,trainY,epochs=200)
pred_train = first_model.predict(trainX)
pred_train= pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-trainY)**2))

plt.plot(pred_train,trainY,"bo")
np.corrcoef(pred_train,trainY) 

#Test on Test Data
first_model.fit(testX,testY,epochs=200)
pred_test = first_model.predict(testX)
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value = np.sqrt(np.mean((pred_test-testY)**2))
plt.plot(pred_test,testY,"bo")
np.corrcoef(pred_train,trainY)
