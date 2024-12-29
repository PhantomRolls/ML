import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
df.drop(['PassengerId','Name','SibSp','Ticket','Cabin','Embarked','Parch'], axis='columns',inplace=True )
target=df.Survived
inputs=df.drop(['Survived'], axis='columns')



dummies=pd.get_dummies(inputs.Sex, dtype=int)
inputs=pd.concat([inputs,dummies],axis='columns')
inputs.drop(['Sex'], axis='columns',inplace=True )
inputs.Age=inputs.Age.fillna(inputs.Age.mean())

#print(inputs.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(inputs,target,test_size=0.2)


