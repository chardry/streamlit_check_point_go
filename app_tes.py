# 1. Installation des packages nécessaires
import joblib
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error

# 2. Importez vos données et effectuez la phase d'exploration de base des données
df = pd.read_csv('dataset/Expresso_churn_dataset.csv')

# Affichage des données
df.head()

# Suppression de certaines colonnes
df.drop(['user_id'], axis=1, inplace=True)

# Détails sur le DataFrame
df.info()

# Quelques statistiques
df.describe().transpose()

# Valeurs dupliqués
df.duplicated().sum()

df.drop_duplicates(inplace=True, keep='first')

# Vérification des valeurs manquantes
df.duplicated().sum()

# Valeurs nulles
df.isnull().sum()

# Récuération des valeurs Objects, int et float
df_obj = df.select_dtypes(include=['object'])
df_int = df.select_dtypes(include=['int64', 'float64'])

# - Données object
df_obj.head()

df_obj.REGION.value_counts()

df_obj.TENURE.value_counts()

df_obj.MRG.value_counts()

df_obj.TOP_PACK.value_counts()

# - Données des int et float
df_int.head()

# Fonction pour remplacer la modalité la plus repandue
def handle_categorical_feature(feature, df):
    df[feature] = df[feature].fillna(df[feature].value_counts().index[0])

for feature in df_obj.columns:
  handle_categorical_feature(feature, df_obj)

for feature in df_int.columns:
  handle_categorical_feature(feature, df_int)

# Concaténation des DataFrames
df_final = pd.concat([df_obj, df_int], axis=1)

ms.bar(df_final)

# Gestion des valeurs abérrantes
# Une visualisation des valeurs abérrantes
def boxplot(data):
    plt.figure(figsize=(16,6))
    sns.boxplot(data=data)
    plt.grid()

boxplot(df_int)

y = df_final['CHURN']
x = df_final.drop(['REGION', 'TENURE', 'MRG', 'TOP_PACK', 'CHURN'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# splitting data with test size of 25%
logreg = LogisticRegression()   #build our logistic model

logreg.fit(X_train, y_train)  #fitting training data

y_pred  = logreg.predict(X_test)    #testing model’s performance

print("Accuracy={:.2f}".format(logreg.score(X_test, y_test)))

def evaluate_model(model):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    rmse_train = mean_squared_error(y_train, train_preds, squared=False)
    rmse_test = mean_squared_error(y_test, test_preds, squared=False)

    return rmse_train, rmse_test

joblib.dump(logreg, "final_model.joblib")
