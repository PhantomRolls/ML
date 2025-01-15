import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from tkinter import Tk, Button
import streamlit as st

# Charger les données
df = pd.read_csv("csv/SPX.csv")


df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month


monthly_returns = df.groupby(['Year', 'Month'])['Adj Close'].agg(['first', 'last'])
monthly_returns['Return (%)'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100

# Réinitialiser l'index pour avoir Year et Month comme colonnes
monthly_returns_reset = monthly_returns.reset_index()
print(monthly_returns)
# Extraire uniquement les colonnes Year, Month et Return (%)
result = monthly_returns_reset[['Year', 'Month', 'Return (%)']]

# Afficher le résultat pour une plage de données choisie avec n et p
n = 73*12   # Début de la plage (index initial)
p = 93*12   # Fin de la plage (index final)
result = result.iloc[n:p].reset_index(drop=True)

print(result)



b = 0 
S = 500000
lig, col = result.shape


# Calculer l'évolution cumulative du portefeuille
cumulative_portfolio = [S]
for i in range(lig):
    cumulative_portfolio.append(
        cumulative_portfolio[-1] * (1 + result['Return (%)'][i] / 100) + b
    )

# Extraire l'année de début
start_year = result['Year'].iloc[0]
start_month = result['Month'].iloc[0]

# Créer les labels pour les abscisses en années et mois
years_months = [f"{start_year + (i // 12)}-{(start_month + i - 1) % 12 + 1:02d}" for i in range(lig)]

def afficher_graphique():
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio, label="Évolution du portefeuille", color='blue')
    plt.title("Évolution du portefeuille avec investissement mensuel constant")
    plt.xlabel("Date (Mois)")
    plt.ylabel("Valeur du portefeuille (€)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Interface Streamlit
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

# Bouton pour afficher ou rétracter le graphique
if st.button("Afficher / Rétracter le graphique"):
    st.session_state.show_graph = not st.session_state.show_graph

# Afficher ou cacher le graphique en fonction de l'état
if st.session_state.show_graph:
    afficher_graphique()



def portefeuille(S):
    for i in range(lig):
        S += b
        S *= (1 + result['Return (%)'][i] / 100)
    return S

def argent_investi(S):
    for i in range(lig):
        S += b
    return S

def rendement(S):
    return portefeuille(S) / argent_investi(S)

def taux_moyen():
    taux_mens=1
    for i in range(lig):
        taux_mens*=1+result['Return (%)'][i] / 100   
    return (taux_mens**(12/lig)-1)*100

def taux_moyen_naïf():
    return np.mean(result['Return (%)'])*12

def portefeuille_moyen(S):
    for i in range(lig):
        S += b
        S *= 1+taux_moyen()/100/12
    return S

def temoin_naïf(S):
    Stot=argent_investi(S)
    for i in range(lig):
        Stot *=(1+taux_moyen_naïf()/12/100)
    return Stot

def temoin_naïf_2(S,taux):
    Stot=argent_investi(S)
    for i in range(lig):
        Stot  *=(1+taux/100/12)
    return Stot


print(f"Rendement moyen annuel : {round(taux_moyen(), 2)}%")
print(f"Portefeuille final : {int(portefeuille(S)):,} €")
print(f"Argent total investi : {int(argent_investi(S)):,} €")
print(f"Rendement global : {round(rendement(S), 2)}")
print(f"Moyenne naïve : {round(taux_moyen_naïf(), 2)}%")
print(f"Portefeuille témoin naïf : {int(temoin_naïf(S)):,} €")
print(f"Nombre d'années analysées : {round(lig / 12, 2)} ans")


