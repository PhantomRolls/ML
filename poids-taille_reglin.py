import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=100
df=pd.read_csv("poids-taille.csv")
df=df.drop(['Genre'], axis='columns')
df['Taille']=(df['Taille']*2.54)
df['Poids']=(df['Poids']*0.453592)
x=df['Taille'].head(p)
y=df['Poids'].head(p)

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

n=len(x)
a=(SP(x,y)-S(x)/n*S(y))/(SP(x,x)-1/n*S(x)**2)
b=1/n*(S(y)-a*S(x))


z=a*x+b
plt.plot(x, z, label=f'Droite: y = {a}x + {b}', color='red')
plt.scatter(x, y)
plt.xlabel('Taille (inches)')
plt.ylabel('Poids (pounds)')
plt.show()

from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(df[['Taille']],df['Poids'])
print(f"a={a} et b={b}")
print(reg.predict([[180]]))
print(a*180+b)