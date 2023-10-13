# -*- coding: utf-8 -*-
"""
@author: 
    
Orlando Maerini
Simona Bevilacqua
Gaetano Albano
Ginevra Biagini
    
"""

import pandas as pd  # Importa la libreria pandas
# Funzione per caricare il dataset originale e fare la pulizia iniziale
def load_and_clean_data(data_file):
    data = pd.read_csv(data_file)  # Legge il file CSV e carica i dati nel dataframe 'data'
    data['DateTime'] = pd.to_datetime(data['DateTime'])  # Converte la colonna 'DateTime' in formato datetime
    data.set_index('DateTime', inplace=True)  # Imposta la colonna 'DateTime' come indice del dataframe
    minute_averages = pd.DataFrame()  # Crea un nuovo dataframe vuoto per le medie dei minuti
    
    # Definizione di una funzione interna per calcolare la media per minuto
    def calculate_minute_average(group):
        return group.resample('1T').mean()  # Calcola la media dei dati ogni minuto

    # Calcola le medie per minuto per le colonne numeriche
    for column in data.columns:
        if data[column].dtype != 'object':
            minute_averages[column] = calculate_minute_average(data[column])

    # Reimposta l'indice e tiene solo le righe con almeno un valore diverso da zero per i dispositivi
    minute_averages.reset_index(inplace=True)
    return minute_averages  # Restituisce il dataframe con le medie per minuto
