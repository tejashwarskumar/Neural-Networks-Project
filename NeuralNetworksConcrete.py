import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

concreteData = pd.read_csv("C:/My Files/Excelr/15 - Neural Networks/Assignment/concrete.csv")
concreteData.head()
concreteData.describe()
concreteData.corr()

concreteData['strength'].describe()
plt.hist(concreteData['strength'])
plt.boxplot(concreteData['strength'])

#remove the outliers
IQR = 46.13 - 23.71
concreteData['strength'].describe()
Range = [(23.71 - 1.5*(IQR)),(46.13 + 1.5*(IQR))]
Range

len(concreteData['strength'])

indexes = [];
for i in range(0,len(concreteData['strength'])):
    if concreteData['strength'][i] > 79.76 :
     indexes.append(i)
     
indexes
concreteData['strength'][[1003]]

concreteData = concreteData[np.bincount(indexes, minlength=len(concreteData)) == 0]
plt.boxplot(concreteData['strength'])
concreteData.isnull().sum()

from sklearn.model_selection import train_test_split
train,test = train_test_split(concreteData,test_size=0.3)
column_names = list(concreteData.columns)
predictors = column_names[0:8]
target = column_names[8]

#Build the model
from keras.models import Sequential
from keras.layers import Dense
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

model = prep_model([8,50,1])
model.fit(np.array(train[predictors]),np.array(train[target]),epochs=90)
pred_train = model.predict(np.array(train[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-train[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,train[target],"bo")
np.corrcoef(pred_train,train[target])

#Test on Test Data
model.fit(np.array(test[predictors]),np.array(test[target]),epochs=90)
pred_test = model.predict(np.array(test[predictors]))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value = np.sqrt(np.mean((pred_test-test[target])**2))
plt.plot(pred_test,test[target],"bo")
np.corrcoef(pred_test,test[target])
