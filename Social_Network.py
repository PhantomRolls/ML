import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




df=pd.read_csv("csv/Social_Network_Ads.csv")
dummies=pd.get_dummies(df['Gender'], dtype=int)

df=pd.concat([df,dummies],axis='columns')
df.drop(['Gender','User ID','Female'],axis='columns', inplace=True)


reg=LogisticRegression()

p=1000
x=df.drop(['Purchased'],axis='columns').to_numpy()[:p]
y=df['Purchased'].to_numpy()[:p]



X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2)



reg = LogisticRegression(max_iter=1000)

reg.fit(X_train,y_train)

input_data=pd.DataFrame([[39,84000,1]])    # columns=['Age','EstimatedSalary','Male']
print(f"prédiction : [[P(0) P(1)]] = {reg.predict_proba(input_data)}")
print("Coefficients:", reg.coef_)
print(f"score : {reg.score(X_test,y_test)}")

