# -*- coding: utf-8 -*-
"""
@author:
Gaetano Albano
"""

import pandas as pd  
import numpy as np

# Funzione per creare la colonna "Device" e identificare i cicli
def create_device_column_and_cycles(data):

    df = data.copy()  # Copia il DataFrame ottenuto

    # Crea la colonna 'Device' combinando i dispositivi attivi separati da virgola
    df['Device'] = df[['wahing_machine', 'dishwasher', 'oven']].apply(
        lambda row: ', '.join([device for device, consumption in row.items() if consumption != 0.0]) if any(consumption != 0.0 for device, consumption in row.items()) else 'altro',
        axis=1
    )

    # Converti la colonna 'DateTime' in formato datetime, gestendo eventuali errori
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')


    # Conta il numero totale di elementi 'altro'
    altro_count = (df['Device'] == 'altro').sum()

    # Mantenimento di 10000 elementi 'altro' corrispettivo di 600000 campioni al senco
    # Verifica se il conteggio di 'altro' è maggiore di 10000
    if altro_count > 10000:
    # Seleziona gli indici degli elementi 'altro' che non sono tra i primi 1500
        indices_to_remove = df.index[df['Device'] == 'altro'][10000:]
    
    # Elimina gli elementi 'altro' che non sono tra i primi 10000
        df.drop(indices_to_remove, inplace=True)
    
    # Converti le fasce orarie da 'DateTime' in minuti
    df['TimeOfDay'] = df['DateTime'].dt.hour * 60 + df['DateTime'].dt.minute

    # Calcola la deviazione standard delle armoniche reali
    df['harmonic_real_std'] = df[['harmonic3_Real', 'harmonic5_Real', 'harmonic7_Real']].std(axis=1)

    # Calcola il prodotto delle armoniche reali
    df['harmonic_real_product'] = df['harmonic3_Real'] * df['harmonic5_Real'] * df['harmonic7_Real']

    # Calcola la deviazione standard delle armoniche reali
    df['ARVC_real_std'] = df[['ActivePower', 'ReactivePower', 'Voltage','Current']].std(axis=1)

    # Calcola il prodotto delle armoniche reali
    df['ARVC_real_product'] = df['ActivePower'] * df['ReactivePower'] * df['Voltage']* df['Current']
    
    # Calcola la deviazione standard delle armoniche reali
    df['rapp_real_std'] = df[['harmonic_real_product', 'ARVC_real_product']].std(axis=1)
    
    # Calcola il prodotto delle armoniche reali
    df['rapp_real_product'] = df['ARVC_real_std'] * df['harmonic_real_std'] 

    # Seleziona le colonne da mantenere
    colonne_da_mantenere = [
        "DateTime",
        "Device",
        "ActivePower",
        "ReactivePower",
        "Voltage",
        "Current",
        "harmonic1_Real",
        "harmonic1_Imaginary",
        "harmonic3_Real",
        "harmonic3_Imaginary",
        "harmonic5_Real",
        "harmonic5_Imaginary",
        "harmonic7_Real",
        "harmonic7_Imaginary",
        "harmonic_real_std",
        "harmonic_real_product",
        "ARVC_real_std",
        "ARVC_real_product",
        "rapp_real_std",
        "rapp_real_product",
        "TimeOfDay"
    ]

    df = df[colonne_da_mantenere]  # Mantieni solo le colonne selezionate
    
    ##salvataggio dataset per test
    # Ottenere il numero totale di righe nel DataFrame
    total_rows = df.shape[0]

    random_indices = np.random.choice(total_rows, 4000, replace=False)

    # Selezionare le righe corrispondenti agli indici casuali
    random_rows = df.iloc[random_indices]

    # Restituire il DataFrame senza le righe corrispondenti agli indici casuali
    df = df.drop(df.index[random_indices]) 


    return df, random_rows  # Restituisce il dataframe modificato con le nuove colonne e i cicli identificati ed il dataframe per il test
