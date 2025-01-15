import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

df=pd.read_csv("csv/poids-taille.csv")
df=df.drop(['Genre'], axis='columns')
df['Taille']=df['Taille']*2.54
df['Poids']=df['Poids']*0.453592
coefficients = {'a': 0, 'b': 0}

p=100
Poids = df['Poids']
Taille = df['Taille']
xi=Taille.to_numpy()[:p]
yi=Poids.to_numpy()[:p]
x = (xi - xi.mean()) / xi.std()
y = (yi - yi.mean()) / yi.std()

def L(a,b):
    return 1/len(x)*sum((y-a*x+b)**2)


def gradient_descent(x,y):
    a=b=0
    it=1000
    n=len(x)
    learning_rate=0.001
    
    for i in range(it):
        y_predicted=a*x+b
        da=-(2/n)*SP(x,y-y_predicted)
        db=-(2/n)*S(y-y_predicted)
        a=a-learning_rate*da
        b=b-learning_rate*db
    Y=a*x+b    
     
    print(f"a={a} et b={b}")
    plt.plot(x,Y, label=f'Droite: y = {a}x + {b}', color='red')
    plt.scatter(x,y)
    plt.xlabel('Taille (inches)')
    plt.ylabel('Poids (pounds)')
    plt.show()

def gradient_descent_anim(x, y):
    a = b = 0
    learning_rate = .05
    n = len(x)
    it = 100

    fig, ax = plt.subplots() 
    line, = ax.plot([], [], label="Régression linéaire", color='red')
    scatter = ax.scatter(xi, yi, label="Données", color='blue')

    ax.set_xlim(min(xi), max(xi))
    ax.set_ylim(min(yi), max(yi))
    ax.set_xlabel('Taille (cm)')
    ax.set_ylabel('Poids (kg)')
    ax.legend()

    
    def update(frame):
        nonlocal a, b
        y_predicted = a * x + b
        da = -(2 / n) * np.dot(x, y - y_predicted)
        db = -(2 / n) * sum(y - y_predicted)
        a -= learning_rate * da
        b -= learning_rate * db
        
        af=yi.std()*a/xi.std()
        bf=(b-xi.mean()/xi.std()*a)*yi.std()+yi.mean()
        y_line = af * xi + bf
        cost=L(a,b)
        line.set_data(xi, y_line)
        ax.set_title(f"y = {af:.2f}x + {bf:.2f}, cost={cost:.2f}")
        coefficients['a']=float(round(af,2))
        coefficients['b']=float(round(bf,2))
        return line,
    
    ani = FuncAnimation(fig, update, frames=it, interval=10, blit=False)
    plt.show()

gradient_descent_anim(x,y)


import pickle

with open('csv/gradient_coefficients.pkl', 'wb') as f:
    pickle.dump(coefficients, f)




