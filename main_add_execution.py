# -*- coding: utf-8 -*-
"""
@author: 
    
Orlando Maerini
Simona Bevilacqua
Gaetano Albano
Ginevra Biagini
    
"""
import argparse
# Import dei moduli personalizzati
from load_and_clean_data import load_and_clean_data
from create_device_column_and_cycles import create_device_column_and_cycles
from scale_selected_columns import scale_selected_columns
from calc_comp_and_pca import perform_pca_with_best_components
from run_classify_svm_cv import optimize_svm_hyperparameters, train_and_evaluate_svm
from pca_test import apply_pca_for_test
# Parsing degli argomenti --Data da linea di comando
parser = argparse.ArgumentParser(description='Process a CSV Data.')
parser.add_argument('--Data', type=str, default='25day_dataset.csv', help='Nome del Data CSV da utilizzare (default: 25day_dataset.csv)')
args = parser.parse_args()


def main():
    # Caricamento e pulizia dei dati
    setdata_clean = load_and_clean_data(args.Data)
    # Creazione di colonne per i dispositivi e i cicli
    setdata_col, test_data = create_device_column_and_cycles(setdata_clean)
    # Scaling delle colonne selezionate per addestramentro e valid
    setdata_scal = scale_selected_columns(setdata_col)
    # Riduzione della dimensionalità tramite PCA
    setdata_pca=perform_pca_with_best_components(setdata_scal)
    best_params = optimize_svm_hyperparameters(setdata_pca)
    print("Migliori Iperparametri SVM:")
    print(best_params)
    # # Addestramento e valutazione del modello SVM
    train_and_evaluate_svm(setdata_pca, best_params)
    # si effettua lo scaling al dataset per il test
    setdata_test_scal = scale_selected_columns(test_data)
    # si effettua la pca al dataset per il test
    apply_pca_for_test(setdata_test_scal)
    
if __name__ == "__main__":
    main()
