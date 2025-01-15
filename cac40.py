import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("csv/CAC_40.csv")

# Convertir la colonne 'Date' en format datetime
df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
df['Price'] = pd.to_numeric(df['Price'].replace({',': '', '€': ''}, regex=True), errors='coerce')

# Extraire l'année et le mois
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Calculer le rendement mensuel
monthly_returns = df.groupby(['Year', 'Month'])['Price'].agg(['first', 'last'])
monthly_returns['Return (%)'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100

# Réinitialiser l'index pour avoir Year et Month comme colonnes
monthly_returns_reset = monthly_returns.reset_index()

# Extraire uniquement les colonnes Year, Month et Return (%)
result = monthly_returns_reset[['Year', 'Month', 'Return (%)']]

# Afficher le résultat pour une plage de données choisie avec n et p
n = 1*12   # Début de la plage (index initial)
p = 20*12   # Fin de la plage (index final)
result = result.iloc[n:p].reset_index(drop=True)

print(result)

# Tracer le graphique du rendement mensuel
plt.figure(figsize=(12, 6))
plt.plot(result.index, result['Return (%)'], marker='o')
plt.title("Rendement mensuel du S&P 500")
plt.xlabel("Mois (index)")
plt.ylabel("Rendement (%)")
plt.grid(True)
#plt.show()

# Calcul des statistiques
S = 2000  # Capital initial
lig, col = result.shape  # Nombre de lignes dans le DataFrame

def portefeuille():
    r = 0
    for i in range(lig):
        r += S
        r *= (1 + result['Return (%)'][i] / 100)
    return r

def temoin():
    r = 0
    for i in range(lig):
        r += S
    return r

def rendement():
    return portefeuille() / temoin()

print("moyenne :", round(np.mean(result['Return (%)'].to_numpy())*12,2),"%")
print("moyenne effective :", round((rendement()-1)*100*12/lig,2),"%")
print("rendement :", round(rendement(),2))
print("portefeuille :", int(portefeuille()))
print("argent investi :", temoin())
print("nombre d'années :", lig/12)