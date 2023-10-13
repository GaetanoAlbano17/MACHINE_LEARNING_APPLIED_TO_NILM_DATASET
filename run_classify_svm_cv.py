# -*- coding: utf-8 -*-
"""
@author: 
    
Orlando Maerini
Simona Bevilacqua
Gaetano Albano
Ginevra Biagini
    
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay 
import pickle
import pandas as pd
# Funzione per ottimizzare gli iperparametri 
def optimize_svm_hyperparameters(df):
    
    X = df.iloc[:, :-2]  # Seleziona tutte le colonne tranne l'ultima
    y = df['Device']
    # Inizializza il classificatore SVM con una griglia di iperparametri
    svm_classifier = SVC()
    param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 'auto', 'scale', 'auto_deprecated'],
    }

    grid_search = GridSearchCV(svm_classifier, param_grid, cv=90, n_jobs=-1)

    # Ricerca dei migliori iperparametri
    grid_search.fit(X, y)

    # Restituisci i migliori iperparametri trovati
    return grid_search.best_params_

# Funzione per eseguire l'addestramento e validazione (con CrossVall a 10 folf )
def train_and_evaluate_svm(df, best_params):
    
    X = df.iloc[:, :-2]  # Seleziona tutte le colonne tranne l'ultima
    y = df['Device']
    
    # Mappa i valori numerici previsti in y alle classi corrispondenti dishwasher oven
    device_mapping = {
        0: 'altro',
        1: 'washing_machine',
        2: 'oven',
        3: 'dishwasher'
    }
    y = y.map(device_mapping)
    # Inizializza il classificatore SVM con i migliori iperparametri ottimizzati
    svm_classifier = SVC(**best_params)

    # Inizializza la strategia di validazione incrociata StratifiedKFold con 10 fold
    cv = StratifiedKFold(n_splits=90)

    # Esegui la validazione incrociata e ottieni le previsioni
    y_pred = cross_val_predict(svm_classifier, X, y, cv=cv)

    # Addestra il classificatore SVM sui dati di addestramento
    svm_classifier.fit(X, y)
    
    # Salvataggio del modello SVM
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)
    # Calcola il classification report
    
    # Calcola il classification report
    class_report = classification_report(y, y_pred, target_names=[device_mapping[class_name] for class_name in df['Device'].unique()], output_dict=True)
    print(f"Classification Report:\n{class_report}")

    # Estrai il Recall e l'F1-score per ogni classe
    classes = df['Device'].unique()
    recall_scores = [class_report[device_mapping[class_name]]['recall'] for class_name in classes]
    f1_scores = [class_report[device_mapping[class_name]]['f1-score'] for class_name in classes]




    ##salvtaggio classification report
    # Creare un DataFrame con i dati di classification report, F1-score, recall e support
    data = {
        'Classe': [device_mapping[class_name] for class_name in classes],
        'F1-score': f1_scores,
        'Recall': recall_scores,
        'Support': [class_report[device_mapping[class_name]]['support'] for class_name in classes]
    }
    df_results = pd.DataFrame(data)
    # Invertire i nomi 'washing_machine' e 'dishwasher' nella colonna 'Classe'
    df_results['Classe'] = df_results['Classe'].replace({'washing_machine': 'dishwasher', 'dishwasher': 'washing_machine'})
    # Specifica il percorso in cui desideri salvare il file CSV
    file_path = 'Plot/Add$Vall/Classification_Report.csv'  # Modifica il percorso come desideri
    
    # Salva il DataFrame in un file CSV
    df_results.to_csv(file_path, index=False)

    
    # Dizionario di mapping per sostituire le etichette delle classi
    # Plotta la matrice di confusione
    confusion_matrix_display = ConfusionMatrixDisplay.from_estimator(svm_classifier, X, y, display_labels=df['Device'].unique(), cmap='Blues', normalize='true')
    confusion_matrix_display.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.show()


    # Salva il grafico come immagine
    plt.savefig('Plot/Add$Vall/confusion_matrix.png')

