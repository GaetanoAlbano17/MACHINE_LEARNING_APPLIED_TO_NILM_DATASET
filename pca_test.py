# -*- coding: utf-8 -*-
"""
@author: 
Gaetano Albano
"""

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def apply_pca_for_test(data):
    # Leggi il dataframe dal file CSV per il test
    df = data.copy()
    # # Carica il modello PCA
    pca_model = joblib.load('pca_model.pkl')

    # Seleziona le colonne desiderate per la PCA e le classi
    selected_columns = ['DateTime', 'Device', 'ActivePower', 'ReactivePower', 'Voltage', 'Current',
                        'harmonic1_Real', 'harmonic1_Imaginary', 'harmonic3_Real', 'harmonic3_Imaginary',
                        'harmonic5_Real', 'harmonic5_Imaginary', 'harmonic7_Real', 'harmonic7_Imaginary',
                        'harmonic_real_std', 'harmonic_real_product', 'ARVC_real_std', 'ARVC_real_product',
                        'rapp_real_std', 'rapp_real_product', 'TimeOfDay']
    selected_classes = ['washing_machine', 'dishwasher', 'oven', 'other']  # Correzione typo

    # Filtra il dataframe per le classi desiderate
    filtered_df = df[df['Device'].isin(selected_classes)][selected_columns]

    # Gestisci i dati mancanti
    numeric_columns = ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary',
                        'harmonic3_Real', 'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary',
                        'harmonic7_Real', 'harmonic7_Imaginary', 'harmonic_real_std', 'harmonic_real_product',
                        'ARVC_real_std', 'ARVC_real_product', 'rapp_real_std', 'rapp_real_product', 'TimeOfDay']
    filtered_df[numeric_columns] = filtered_df[numeric_columns].fillna(filtered_df[numeric_columns].mean())

    # Effettua la codifica delle classi 'Device'
    label_encoder = LabelEncoder()
    filtered_df['Device'] = label_encoder.fit_transform(filtered_df['Device'])

    # Standardizzazione delle features numeriche
    scaler = StandardScaler()
    filtered_df[numeric_columns] = scaler.fit_transform(filtered_df[numeric_columns])

    # Applica la PCA utilizzando il modello salvato
    pca_result = pca_model.transform(filtered_df[numeric_columns])

    # Crea un nuovo DataFrame con le componenti PCA
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(pca_result.shape[1])])

    # Aggiunta colonna Device al nuovo dataframe
    pca_df['Device'] = filtered_df['Device']

    # Estrai e ordina la colonna 'DateTime' dal dataframe originale df
    datetime_column = df[df['Device'].isin(selected_classes)]['DateTime'].sort_values()

    # Crea una nuova colonna 'DateTime' nel dataframe pca_df con i valori ordinati
    pca_df['DateTime'] = datetime_column.values

    # salva il nuovo dataset per test con pca
    pca_df.to_csv("Test/test_pca.csv", index=False)
