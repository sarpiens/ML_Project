#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: evaluation.py

Description:

Código para evaluar el modelo utilizando los datos de prueba de la carpeta `data/test` y generar métricas de evaluación.

Proyecto individual curso Data Science - The Bridge y Fundación Universia (2026-09).

Author: Samuel PE
Date: 27 Noviembre 2026

Version: 1.0.0
"""
#Importar librerias 
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score

# Atributos del programa y constantes
PROGRAM_NAME = "evaluation.py"
__version__ = "1.0.0"

# Paths relativos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data"/ "test" / "online_shoppers_intention_processed_test.csv"
TRAIN_RESULTS_PATH = PROJECT_ROOT / "models"/ "final_propensity_model.joblib"
TEST_RESULTS_PATH = PROJECT_ROOT / "models"/ "final_propensity_model_evaluation.joblib"

#Variables
TARGET = "Revenue"

#Definimos métricas model test
scoring_test = {
    "Avg_Precision": average_precision_score,
    "ROC_AUC": roc_auc_score,
    "Brier_Score": brier_score_loss,
}

# Funciones del programa
def carga_test_set():
    #Mensaje del paso actual
    print("\n1. Carga y Preparación del Test Set:")
    print(TEST_PATH)
    return pd.read_csv(TEST_PATH, sep=",", index_col=0)

def carga_modelo_resultados():
    #Mensaje del paso actual
    print("\n2. Carga del Modelo Final y Resultados:")
    print(TRAIN_RESULTS_PATH)
    return joblib.load(TRAIN_RESULTS_PATH)

def get_test_model_metrics_and_preds(scoring_test, model_pipeline, X_test_set, y_test_set, train_cv_metrics):
    # 0. Mensaje del paso
    print("\n3.1. Test del modelo LGBMClassifier:")
    # 1. Hacemos testing del modelo- obtenemos probabilidades
    test_preds = model_pipeline.predict_proba(X_test_set)[:, 1]
    #2. Generamos las métricas de test
    metricas_globales_test = {
        nombre: round(funcion(y_test_set, test_preds), 4)
        for nombre, funcion in scoring_test.items()
        }
    # A partir del diccionario generamos pandas df
    df_test = pd.DataFrame({"set": "Test",
    "metric": list(metricas_globales_test.keys()),
    "score": list(metricas_globales_test.values())
    })
    #3. Mostrar métricas de train cv
    ## Imprimir tabla train cv en terminal
    print("\nMétricas de Train CV:\n")
    print(train_cv_metrics.to_string(index=False))
    #4. Mostrar métricas de test
    ## Imprimir tabla test en terminal
    print("\nMétricas de Test:\n")
    print(df_test.to_string(index=False))
    #Devolvemos resultados
    return df_test, test_preds

def get_test_threshold_metrics(thresholds, df_oof_results, test_preds, y_test_set):
    #0.Mensajes iniciales
    print("\n3.2. Obtención de métricas basadas en Test Set para Audiencias:")
    print("\nAudiencias (Umbrales):\n")
    for segmento, valor in thresholds.items():
        print(f"- {segmento} ({valor})")
    # 1.Obtenemos métricas
    ##Calculamos tasa_base(prop P en el set)
    tasa_base_prop_P_set = y_test_set.mean() #Como son 1 y 0, hacer la media es equivalente a sacar proporcion P en el set
    ## Definimos lista resultados
    resultados_metricas_test = []
    # Obtenemos métricas para cada umbral del dict thresholds {nombre audiencia:umbral}
    for nombre, th in thresholds.items():
        #Calculos
        preds_bin = (test_preds >= th).astype(int) #Pasamos propensity scores predichos a binaria (en base a ese umbral th)
        leads_fp = int(np.sum((y_test_set == 0) & (preds_bin == 1))) #Calculamos total leads (FP para esa audiencia)
        total_sel = int(np.sum(preds_bin == 1)) #Calculamos Total_Predicted_Positives
        prec = precision_score(y_test_set, preds_bin) #Calculamos precision score
        #Guardamos
        resultados_metricas_test.append({
            "Audiencia": nombre,
            "Set": "Test",
            "Umbral_Score": round(th, 4),
            "Precision": round(prec, 4),
            "Recall": round(recall_score(y_test_set, preds_bin), 4),
            "Lift (Precision / Prop_P)":round(prec / tasa_base_prop_P_set, 2),
            "Leads_Retargeting_FP": leads_fp,
            "Total_Predicted_Positives": total_sel,
            "Perc_LeadsFP_Predicted_Positives": round((leads_fp / total_sel)*100, 2),
            "Perc_LeadsFP_Total_Users": round((leads_fp / len(y_test_set))*100, 2),
            })
    # Convertir a dataframe resultados y dict de umbrales
    df_umbrales_test = pd.DataFrame(resultados_metricas_test)
    #2. Mostrar métricas de train cv
    ## Imprimir tabla del train cv en terminal
    print("\nMétricas de Train CV:\n")
    print(df_oof_results.to_string(index=False))
    #3. Mostrar métricas de test
    ## Imprimir tabla test en terminal
    print("\nMétricas de Test:\n")
    print(df_umbrales_test.to_string(index=False))
    # Devolver resultados
    return df_umbrales_test

def save_test_results(df_test, df_umbrales_test):
    # Guardamos resultados
    print("\n3.3. Guardando Resultados de Fase Evaluación (Evaluation):")
    print(TEST_RESULTS_PATH)

    # Empaquetar en un dict test_results
    test_results = {
            "df_model_test_metrics": df_test,
            "df_thresholds_test_metrics":df_umbrales_test,
        }
    # Guardar resultados
    joblib.dump(test_results, TEST_RESULTS_PATH)

# Programa principal
def main():
    ##0. Mensaje nombre y version programa
    print(f"\n{PROGRAM_NAME} v {__version__}")
    # 1. Preparamos Test set
    ## 1.1 Carga del test set
    test_set = carga_test_set()
    ## 1.2. Separamos X e y para test set
    print("\nSeparamos Test Set en X e y ...")
    X_test_set = test_set.drop(columns=[TARGET])
    y_test_set = test_set[TARGET]
    # 2. Cargamos modelo y resultados
    train_results = carga_modelo_resultados()
    # Y lo separamos para operar
    model_pipeline = train_results["pipeline"] # Incluye tu ColumnTransformer + LightGBM entrenados con train
    train_cv_metrics = train_results["train_cv_metrics"]
    thresholds = train_results["thresholds"]
    df_oof_results = train_results["df_oof_results"]
    ## 3. Fase de Evaluation
    print("\n3. Fase de Evaluación (Evaluation):")
    ## 3.1. Obtenemos métricas test del modelo y predicciones (mostrando métricas Train CV vs Test)
    df_test, test_preds = get_test_model_metrics_and_preds(scoring_test, model_pipeline, X_test_set, y_test_set, train_cv_metrics)
    ## 3.2. Obtenemos métricas test de los umbrales
    df_umbrales_test = get_test_threshold_metrics(thresholds, df_oof_results, test_preds, y_test_set)
    ## 3.3. Guardar resultados de test
    save_test_results(df_test, df_umbrales_test)

if __name__ == "__main__":
    main()