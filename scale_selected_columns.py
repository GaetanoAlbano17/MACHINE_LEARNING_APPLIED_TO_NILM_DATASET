# -*- coding: utf-8 -*-
"""
@author: 
    
Orlando Maerini
Simona Bevilacqua
Gaetano Albano
Ginevra Biagini
    
"""

from sklearn.preprocessing import MaxAbsScaler  # Importa la funzione MaxAbsScaler da sklearn.preprocessing
# Funzione per scalare le colonne desiderate
def scale_selected_columns(data):
    df = data.copy()  # Crea una copia del dataframe di input
    columns_to_scale = ['ActivePower', 'ReactivePower', 'Voltage', 'Current', 'harmonic1_Real', 'harmonic1_Imaginary', 'harmonic3_Real', 'harmonic3_Imaginary', 'harmonic5_Real', 'harmonic5_Imaginary', 'harmonic7_Real', 'harmonic7_Imaginary', 'harmonic_real_std', 'harmonic_real_product', 'ARVC_real_std', 'ARVC_real_product', 'rapp_real_std', 'rapp_real_product', 'TimeOfDay']
    # Crea un oggetto scaler di tipo MaxAbsScaler
    scaler = MaxAbsScaler()
    # Applica la trasformazione di scaling MaxAbs a colonne selezionate
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df  # Restituisce il dataframe scalato
