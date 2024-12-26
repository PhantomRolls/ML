import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv("reglin-dataset.csv")

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

n=len(df["taille"])
a=(SP(df["taille"],df["poids"])-S(df["taille"])/n*S(df["poids"]))/(SP(df["taille"],df["taille"])-1/n*S(df["taille"])**2)
b=1/n*(S(df["poids"])-a*S(df["taille"]))
        
def f(x):
    return a*x+b

x=df['taille']
y=a*x+b
plt.plot(x, y, label=f'Droite: y = {a}x + {b}', color='red')
plt.scatter(df['taille'], df['poids'])
plt.show()