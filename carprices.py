import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv("csv/carprices.csv")

dummies=pd.get_dummies(df['Car Model'], dtype=int)

df=pd.concat([df,dummies],axis='columns')
df.drop(['Car Model','Mercedez Benz C class'],axis='columns',inplace=True)

X=df.drop(['Sell Price($)'],axis='columns')
y=df['Sell Price($)']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)

input_data=pd.DataFrame([[45000,4,0,0]], columns=['Mileage','Age(yrs)','Audi A5','BMW X5'])    
print(f"prédiction : {reg.predict(input_data)}")

print(f"valeurs : {reg.predict(X_test).astype(int)}")
print(f"prédictions : {y_test.to_numpy()}")
print(f"score : {reg.score(X_test,y_test)}")


ones_column = np.ones((X_train.shape[0], 1))
X_t=np.concatenate((ones_column,X_train.to_numpy()),axis=1)

B=np.linalg.inv(X_t.T@X_t)@X_t.T@y_train.to_numpy()
print("B calculé :",B)
print("B  du modèle :",reg.intercept_,reg.coef_)
