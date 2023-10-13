# -*- coding: utf-8 -*-
"""
@author: 
    
Orlando Maerini
Simona Bevilacqua
Gaetano Albano
Ginevra Biagini
    
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Funzione che ottiene il numero di componenti ideale per la PCA 
# che verra eseguita con tali componenti
def perform_pca_with_best_components(df):

    # Seleziona le colonne desiderate per la PCA e le classi
    selected_columns = ['DateTime', 'Device', 'ActivePower', 'ReactivePower', 'Voltage', 'Current','harmonic1_Real','harmonic1_Imaginary','harmonic3_Real','harmonic3_Imaginary','harmonic5_Real','harmonic5_Imaginary','harmonic7_Real', 'harmonic7_Imaginary','harmonic_real_std','harmonic_real_product','ARVC_real_std','ARVC_real_product','rapp_real_std','rapp_real_product','TimeOfDay']
    selected_classes = ['wahing_machine', 'dishwasher', 'oven','altro']

    # Filtra il dataframe per le classi desiderate
    filtered_df = df[df['Device'].isin(selected_classes)][selected_columns]

    # Gestisci i dati mancanti
    numeric_columns = ['ActivePower', 'ReactivePower', 'Voltage', 'Current','harmonic1_Real','harmonic1_Imaginary','harmonic3_Real','harmonic3_Imaginary','harmonic5_Real','harmonic5_Imaginary','harmonic7_Real', 'harmonic7_Imaginary','harmonic_real_std','harmonic_real_product','ARVC_real_std','ARVC_real_product','rapp_real_std','rapp_real_product','TimeOfDay']
    filtered_df[numeric_columns] = filtered_df[numeric_columns].fillna(filtered_df[numeric_columns].mean())

    # Effettua la codifica delle classi 'Device'
    label_encoder = LabelEncoder()
    filtered_df['Device'] = label_encoder.fit_transform(filtered_df['Device'])

    # Standardizzazione delle features numeriche
    scaler = StandardScaler()
    filtered_df[numeric_columns] = scaler.fit_transform(filtered_df[numeric_columns])

    # Calcola il numero ideale di componenti principali
    pca = PCA()
    pca.fit(filtered_df[numeric_columns])

    # calcolo componendi grafiche
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Trova il numero ottimale di componenti principali
    optimal_num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print("Miglior numero componenti PCA:")
    print(optimal_num_components)
    # Esegui la PCA con il numero ottimale di componenti principali
    pca = PCA(n_components=optimal_num_components)
    pca_result = pca.fit_transform(filtered_df[numeric_columns])
    # Creazione di un nuovo dataframe con le componenti principali
    pca_columns = [f'PC{i + 1}' for i in range(optimal_num_components)]
    pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)

    # Visualizza la varianza spiegata cumulativa
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Numero di componenti')
    plt.ylabel('Varianza spiegata cumulativa')
    plt.title('Varianza spiegata cumulativa vs Numero di componenti')
    plt.savefig('Plot/Add$Vall/Varianza spiegata cumulativa vs Numero di componenti.png')
    plt.show()
    # Visualizza la rappresentazione delle componenti principali (ad esempio, prime due componenti)
    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=filtered_df['Device'], cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Rappresentazione delle componenti principali (PC1 vs PC2)')
    plt.colorbar(label='Device')
    plt.savefig('Plot/Add$Vall/Rappresentazione(PC1 vs PC2).png')
    plt.show()

    # Plot tridimensionale
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Aggiunge l'asse 3D al grafico

    # Crea il plot tridimensionale
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=filtered_df['Device'], cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Rappresentazione delle componenti principali (PC1 vs PC2 vs PC3)')

    # Salva il plot tridimensionale
    plt.savefig('Plot/Add$Vall/Rappresentazione(PC1 vs PC2 vs PC3).png')
    plt.show()
    # Aggiunta colonna Device al nuovo dataframe
    pca_df['Device'] = filtered_df['Device'] = label_encoder.fit_transform(filtered_df['Device'])
    
    # Estrai e ordina la colonna 'DateTime' dal dataframe originale df
    datetime_column = df[df['Device'].isin(selected_classes)]['DateTime'].sort_values()

    # Crea una nuova colonna 'DateTime' nel dataframe pca_df con i valori ordinati
    pca_df['DateTime'] = datetime_column.values
    
    # Salva il modello PCA in un file
    with open('pca_model.pkl', 'wb') as file:
        joblib.dump(pca, file)
        
    return pca_df