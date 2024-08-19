import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Informations
st.title('Prédiction si un client va quitter ou non Expresso ?')
st.subheader('Application realisee dans le cadre de GoMycode')
st.markdown('Cette application utilise Modele de ML pour prédire si un client va qui ou non Expresso')

# Chargement du modèle
model = joblib.load(filename='final_model.joblib')

# Définition d'une fonction d'inférence
def inference(MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
              FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO, ZONE1, ZONE2, REGULARITY, FREQ_TOP_PACK):
    new_data = np.array([
        MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
        FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
        ZONE1, ZONE2, REGULARITY, FREQ_TOP_PACK
    ])

    pred = model.predict(new_data.reshape(1, -1))
    return pred


# Création du formulaire
st.text('Les variables nécessaires pour Prédire le résultat')
MONTANT = st.number_input(label='MONTANT', value=2700.0)
FREQUENCE_RECH = st.number_input(label='FREQUENCE_RECH', value=6.0)
REVENUE = st.number_input(label='REVENUE', value=2705.0)
ARPU_SEGMENT = st.number_input(label='ARPU_SEGMENT', value=902.0)
FREQUENCE = st.number_input(label='FREQUENCE', value=8.0)
DATA_VOLUME = st.number_input(label='DATA_VOLUME', value=1.0)
ON_NET = st.number_input(label='ON_NET', value=19.0)
ORANGE = st.number_input(label='ORANGE', value=18.0)
TIGO = st.number_input(label='TIGO', value=1.0)
ZONE1 = st.number_input(label='ZONE1', value=0.0)
ZONE2 = st.number_input(label='ZONE2', value=0.0)
REGULARITY = st.number_input(label='REGULARITY', value=40.0)
FREQ_TOP_PACK = st.number_input(label='FREQ_TOP_PACK', value=3.0)

# Création du Bouton "Predict" qui retourne la prédiction du modèle
if st.button('Predict'):
    prediction = inference(
        MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
        FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
        ZONE1, ZONE2, REGULARITY, FREQ_TOP_PACK
    )
    st.success(prediction)