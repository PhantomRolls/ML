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

p=100
x=df.drop(['Purchased'],axis='columns').to_numpy()[:p]
y=df['Purchased'].to_numpy()[:p]

X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2)



def sigma(x):
    return 1/(1+np.exp(-x))

ones_column = np.ones((X_train.shape[0], 1))
X=np.concatenate((ones_column,X_train),axis=1)

'''
def gradient_descent(x,y):
    
    
    n=len(X)
    k=len(X[0])
    b= np.random.randint(1, 10, size=(k, 1))
    print(b.reshape(-1,1))
    print(X[0])
    
    grad=[]
    for i in range(n):
        dij=0
        for j in range(k):
            dij+=-X[i][j]*sigma(X[i]@b.reshape(-1,1))@(1-sigma(X[i]@b.reshape(-1,1)))
            print(-X[i][j]*sigma(X[i]@b.reshape(-1,1))@(1-sigma(X[i]@b.reshape(-1,1))))
            
        
        grad.append([dij])
        return(grad)
    
        
      
print(gradient_descent(X_train,y_train))'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Appliquer une normalisation des données sur X_train et X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 1. Sigmoïde : fonction d'activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Fonction de coût (Log Loss)
def cost_function(X, y, theta):
    m = len(y)  # Nombre d'exemples
    h = sigmoid(X.dot(theta))  # Calcul de la prédiction
    epsilon = 1e-5  # Éviter log(0)
    cost = -(1/m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost

# 3. Descente de gradient
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)  # Nombre d'exemples
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(theta))  # Calcul de la prédiction
        gradient = (1/m) * X.T.dot(h - y)  # Calcul du gradient
        theta = theta - learning_rate * gradient  # Mise à jour des paramètres
        
        # Sauvegarder la valeur de la fonction de coût pour chaque itération
        cost_history.append(cost_function(X, y, theta))

    return theta, cost_history

# 4. Prédiction
def predict(X, theta):
    probability = sigmoid(X.dot(theta))
    return [1 if p >= 0.5 else 0 for p in probability]

# 5. Entraînement du modèle
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    # Initialisation des paramètres
    theta = np.zeros(X.shape[1])  # Vecteur de poids initialisé à zéro
    # Appliquer la descente de gradient
    theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
    return theta, cost_history

# 6. Fonction principale pour tester la régression logistique
if __name__ == "__main__":
    # X_train, y_train, X_test, y_test sont déjà définis dans votre code

    # 7. Entraîner le modèle avec les données d'entraînement
    theta, cost_history = logistic_regression(X_train, y_train, learning_rate=0.1, iterations=1000)

    # Affichage des coefficients appris
    print("Paramètres appris (theta) :", theta)

    # 8. Prédictions sur les données d'entraînement
    predictions_train = predict(X_train, theta)
    print("Prédictions sur les données d'entraînement :", predictions_train)

    # 9. Prédictions sur les données de test
    predictions_test = predict(X_test, theta)
    print("Prédictions sur les données de test :", predictions_test)

    # 10. Affichage de l'évolution du coût
    plt.plot(cost_history)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Fonction de coût")
    plt.title("Evolution du coût (log loss) au fil des itérations")
    #plt.show()

    # 11. Calcul du score sur les données de test (accuracy)
    accuracy = np.mean(predictions_test == y_test)
    print(f"Précision sur les données de test : {accuracy * 100:.2f}%")
    
    input_data = np.array([1, 14000, 1, 29])  # Nouvelle donnée (avec "1" ajouté pour le biais)
    prediction_proba = predict(input_data, theta)  # Calcul de la probabilité d'appartenance à la classe 1

    # Affichage de la probabilité d'appartenance à la classe 1
    print(f"Probabilité d'appartenir à la classe 1 pour la donnée {input_data} : {prediction_proba}")
