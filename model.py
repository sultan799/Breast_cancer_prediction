#import necessary libraries
#import libraries
import numpy as np
import pandas as pd
import sklearn.datasets
#import matplotlib.pyplot as plt
#import seaborn as sbn
import pickle

#collecting dataset
breast_cancer_cases=sklearn.datasets.load_breast_cancer()
print(breast_cancer_cases)

X=breast_cancer_cases.data#storing input features columns in X
Y=breast_cancer_cases.target#stroring label/output column in Y
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

#import data to pandas dataframe 
data=pd.DataFrame(breast_cancer_cases.data,columns=breast_cancer_cases.feature_names)
data['result']=breast_cancer_cases.target
data.head()
data.describe()
#print(data['result'].value_counts())


data.groupby('result').mean()

#splitting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X.shape,Y.shape
#print(Y.mean(),Y_train.mean(),Y_test.mean())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
#print(Y.mean(),Y_train.mean(),Y_test.mean())
#print(X.mean(),X_train.mean(),X_test.mean())

#Model building
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)

#Model Evaluation
from sklearn.metrics import accuracy_score
predict=model.predict(X_train)
accuracy_on_train_data=accuracy_score(Y_train,predict)
#print('accuracy_on_train_data is ',accuracy_on_train_data)

prediction_on_test_data=model.predict(X_test)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
#print('accuracy_on_test_data is ',accuracy_on_test_data)

#plotting on graph
#plt.plot(X_test,prediction_on_test_data)
#plt.xlabel('features')
#plt.ylabel('result')
#plt.show()

with open('model.pkl','wb') as f:
  pickle.dump(model,f)
with open('model.pkl','rb') as f:
  model=pickle.load(f)
