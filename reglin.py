import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv("poids-taille.csv")
df=df.drop(['Genre'], axis='columns')
def SP(X,Y):
    r=0
    for i in range(len(X)):
        r+=X[i]*Y[i]
    return r
def S(X):
    r=0
    for i in range(len(X)):
        r+=X[i]
    return r

n=len(df["Taille"])
a=(SP(df["Taille"],df["Poids"])-S(df["Taille"])/n*S(df["Poids"]))/(SP(df["Taille"],df["Taille"])-1/n*S(df["Taille"])**2)
b=1/n*(S(df["Poids"])-a*S(df["Taille"]))
        
def f(x):
    return a*x+b

x=df['Taille']
y=a*x+b
plt.plot(x, y, label=f'Droite: y = {a}x + {b}', color='red')
plt.scatter(df['Taille'], df['Poids'])
plt.xlabel('Taille')
plt.ylabel('Poids')
plt.show()