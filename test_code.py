# -*- coding: utf-8 -*-
"""
@author: 
Gaetano Albano
"""
import pandas as pd
import joblib
from sklearn.metrics import recall_score, f1_score
import numpy as np


# Carica il modello SVM
svm_model = joblib.load('svm_model.pkl')



#### organizzazione file originale per test 


# Carica il dataframe originale
df_original = pd.read_csv('25day_dataset.csv')
df_original['Device'] = df_original[['wahing_machine', 'dishwasher', 'oven']].apply(
    lambda row: ', '.join([device for device, consumption in row.items() if consumption != 0.0]) if any(consumption != 0.0 for device, consumption in row.items()) else 'altro',
    axis=1
)
# Invertire i nomi 'wahing_machine' e 'washing_machine' nella colonna 'Device' per correggere errore mapping
df_original['Device'] = df_original['Device'].replace({'wahing_machine': 'washing_machine'})
#formato datetime
df_original['DateTime'] = pd.to_datetime(df_original['DateTime'])


####  Predizione

# Carica il dataframe con PCA
df = pd.read_csv('Test/test_pca.csv')
# corrrezione errore mapping
mapping = {
    0: 'altro',
    1: 'dishwasher',
    2: 'oven',
    3: 'washing_machine'
}

# Applica la mappatura alla colonna 'Device'
df['Device'] = df['Device'].map(mapping)

# Seleziona solo le prime 6 colonne come features
features = df.iloc[:, :-2]  # Escludi le ultime due colonne (DateTime  & Device che serviranno per studiare recall e F1 dei dati predetti sulla PCA)

# Effettua la previsione usando il modello SVM
predictions = svm_model.predict(features)

#### analisi Previsioni PCA

# Crea un nuovo DataFrame con le previsioni e i nomi delle features
predictions_df = pd.DataFrame(predictions, columns=['device_predetto'], index=features.index)
# Invertire i nomi 'washing_machine' e 'dishwasher' nella colonna 'Classe'
predictions_df['device_predetto'] = predictions_df['device_predetto'].replace({'washing_machine': 'dishwasher', 'dishwasher': 'washing_machine'})

# Concatena il DataFrame delle previsioni con il DataFrame originale
df_with_predictions = pd.concat([df, predictions_df], axis=1)

# Calcola il recall e l'F1-score per ogni classe
labels = ['altro', 'washing_machine', 'oven', 'dishwasher']
recall_per_class = recall_score(df_with_predictions['Device'], df_with_predictions['device_predetto'], labels=labels, average=None)
f1score_per_class = f1_score(df_with_predictions['Device'], df_with_predictions['device_predetto'], labels=labels, average=None)

# Approssima i dati a due decimali
recall_per_class = recall_per_class.round(2)
f1score_per_class = f1score_per_class.round(2)

# Crea un DataFrame con i risultati
results_df = pd.DataFrame({'Classe': labels, 'Recall': recall_per_class, 'F1-score': f1score_per_class})

# Stampa il DataFrame
print("Risultati di Recall e F1-score per ogni classe:")
print(results_df)

# Salva il DataFrame in un file CSV
results_df.to_csv('Test/risultati_recall_f1_PCA_data.csv', index=False)
 

#### analisi Previsioni sul File Originale 
   
# Ordina i dati in base alla colonna 'DateTime'
df_with_predictions['DateTime'] = pd.to_datetime(df_with_predictions['DateTime'])
df_with_predictions = df_with_predictions.sort_values(by='DateTime')

# Replica le righe per ogni blocco di 60 righe
df_with_predictions = df_with_predictions.loc[df_with_predictions.index.repeat(60)].reset_index(drop=True)

# Calcola i secondi da 0 a 59 per ogni blocco di 60 righe
df_with_predictions['Seconds'] = np.tile(np.arange(60), len(df_with_predictions) // 60 + 1)[:len(df_with_predictions)]


##concatenazione tramite DateTime

# Merge basato sulla colonna 'DateTime'
merged_df = pd.merge(df_original, df_with_predictions[['DateTime', 'device_predetto']],
                     on='DateTime', how='left')

# Rinomina la colonna 'device_predetto' nel dataframe originale
merged_df.rename(columns={'device_predetto': 'device_predetto'}, inplace=True)
# Elimina le righe con 'device_predetto' mancante (NaN)
merged_df.dropna(subset=['device_predetto'], inplace=True)

# Calcola il recall e l'F1-score per ogni classe
labels = ['altro', 'washing_machine', 'oven', 'dishwasher']
recall_per_class = recall_score(merged_df['Device'], merged_df['device_predetto'], labels=labels, average=None)
f1score_per_class = f1_score(merged_df['Device'], merged_df['device_predetto'], labels=labels, average=None)

# Approssima i dati a due decimali
recall_per_class = recall_per_class.round(2)
f1score_per_class = f1score_per_class.round(2)

# Crea un DataFrame con i risultati
results_df = pd.DataFrame({'Classe': labels, 'Recall': recall_per_class, 'F1-score': f1score_per_class})

# Stampa il DataFrame
print("Risultati di Recall e F1-score per ogni classe:")
print(results_df)

# Salva il DataFrame in un file CSV
results_df.to_csv('Test/risultati_recall_f1_Orig_data.csv', index=False)
merged_df.to_csv('Test/Origmerged_df.csv', index=False)




