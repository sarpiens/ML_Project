#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: training.py

Description:

Código para entrenar y guardar el modelo entrenado con el input de los datos de la carpeta `data/processed` 
y guardar los datasets de `data/train` y `data/test`.

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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier

# Atributos del programa y constantes
PROGRAM_NAME = "training.py"
__version__ = "1.0.0"

# Paths relativos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data"/ "processed" / "online_shoppers_intention_processed.csv"
TRAIN_PATH = PROJECT_ROOT / "data"/ "train" / "online_shoppers_intention_processed_train.csv"
TEST_PATH = PROJECT_ROOT / "data"/ "test" / "online_shoppers_intention_processed_test.csv"
TRAIN_RESULTS_PATH = PROJECT_ROOT / "models"/ "final_propensity_model.joblib"

# Hiperparámetros ganadores del modelo LGBMClassifier con 
BEST_PARAMS = {'bagging_fraction': 1,
 'bagging_freq': 1,
 'feature_fraction': 0.5,
 'learning_rate': 0.05,
 'max_bin': 255,
 'max_depth': 6,
 'n_estimators': 100,
 'reg_lambda': 1,
 'random_state':42, 
 'verbose': -100, 
 'n_jobs':1,
 }

# Definimos metricas CV Train
scoring = {
    "Avg_Precision": "average_precision",
    "ROC_AUC": "roc_auc",
    "Brier_Score": "neg_brier_score",
}
 
# Definimos las audiencias top para el calculo de umbrales del OOF
audiencias_top_pct = [5, 15, 30]

# Variables seleccionadas después del EDA separadas por tratamiento
## Target 
TARGET = "Revenue"
## Features Numéricas
cols_log_scaler = ['Administrative', 'Informational', 'ProductRelated','Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
cols_scaler_only = ['ProductRelated_Ratio', 'ProductRelated_Duration_Ratio','SpecialDay', 'ExitRates']
## Features Categóricas
cols_std_cat = ["Month", "VisitorType"]
col_cat_TrafficType = ['TrafficType']
## All features
features_finales = cols_log_scaler + cols_scaler_only + cols_std_cat + col_cat_TrafficType
## All columns
columnas_finales = features_finales + [TARGET]

# Preparación del transformer_base
#####################################################################
# 1. Variables numéricas / Crear sub-pipelines para las numéricas
pipeline_log_scaler = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #Añadimos imputer por robustez operativa futura
    ('log1p', FunctionTransformer(func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")),
    ('scaler', StandardScaler())
])
pipeline_num_scaler = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #Añadimos imputer por robustez operativa futura
    ('scaler', StandardScaler())
])

# 2. Variables categóricas / Crear sub-pipelines para las categoricas
## En este caso no necesitamos imputar (NAs y desconocidos pondran columnas asociadas a cero)
##Pipeline para cat_TrafficType
### Definimos el pipeline añadiendo tambien `min_frequency=0.01` agrupará los canales conocidos que son poco frecuentes (< 1%) en una columna TrafficType_infrequent_sklearn
pipeline_cat_TrafficType = Pipeline([
    ("to_str", FunctionTransformer(np.asarray, kw_args={'dtype': str}, feature_names_out="one-to-one")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore", min_frequency=0.01))
])

# 3. Unificar transformaciones en un ColumnTransformer
transformer_base = ColumnTransformer(
    transformers=[
        ('log_scaler', pipeline_log_scaler, cols_log_scaler),
        ('num_scaler', pipeline_num_scaler, cols_scaler_only),
        ('cat_std', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cols_std_cat),
        ('cat_TrafficType',pipeline_cat_TrafficType, col_cat_TrafficType)
    ],
    remainder='drop', # Quitamos el resto de columnas
    verbose_feature_names_out=False # Mantiene los nombres originales de las columnas
).set_output(transform='pandas') # Devuelve un DataFrame de pandas en lugar de un array numpy
#####################################################################

# Funciones del programa
def carga_processed_dataset():
    #Mensaje del paso actual
    print("\n1. Carga del processed dataset:")
    print(PROCESSED_PATH)
    return pd.read_csv(PROCESSED_PATH, sep=",", index_col=0)

def split_train_test(dataset):
    #Mensaje del paso actual
    print("\n2. Split Train-Test del processed dataset:")
    print(f'Parámetros: test_size=0.2, random_state=42, stratify=dataset[{TARGET}]')
    #Además, también tiene shuffle=True por defecto
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[TARGET])
    #Mensaje del paso actual(2)
    print("\nGuardando Split Train-Test sets:")
    print(TRAIN_PATH)
    train_set.to_csv(TRAIN_PATH, sep=",")
    print(TEST_PATH)
    test_set.to_csv(TEST_PATH, sep=",")
    #Devolver sets
    return train_set, test_set

def train_model(X_train_set, y_train_set, transformer_base):
    # 0. Mensaje del paso
    print("\n3.1. Entrenamiento del modelo LGBMClassifier:")
    print("Parámetros:")
    for param, value in BEST_PARAMS.items():
        print(f" - {param}: {value}")
    # 1. Definimos el pipeline completo
    pipeline_final = Pipeline(steps=[("preprocessor", clone(transformer_base)),
                                     ("modelo", LGBMClassifier(**BEST_PARAMS))
                                     ])
    # 2. Entrenamiento solo con train
    pipeline_final.fit(X_train_set, y_train_set)
    # 3. Devolvemos el resultado
    return pipeline_final

def get_cv_metrics(X_train_set, y_train_set, pipeline_final, scoring, n_splits=5):
    # 0. Mensaje del paso
    print(f"\n3.2. Obtención de métricas Train CV={n_splits}:\n")
    # Realizamos cross_validate basado en train con metricas(scoring)
    cv_results = cross_validate(
      estimator=clone(pipeline_final),
      X=X_train_set,
      y=y_train_set,
      cv=n_splits,
      scoring=scoring,
      n_jobs=-1,
    )
    #Preparamos los resultados
    rows_results = []
    for metric_name, scorer in scoring.items():
        scores = cv_results[f"test_{metric_name}"]
        mean_val = float(np.mean(scores))
        # Invertir el signo para métricas neg_* (como neg_brier_score)
        if isinstance(scorer, str) and scorer.startswith("neg_"):
            mean_val = -mean_val
        #Guardar resultados
        rows_results.append({
            "set": f"Train:CV={n_splits}",
            "metric": metric_name,
            "mean": round(mean_val, 4),
            "std": round(float(np.std(scores)), 4),
    })
    #Guardamos como df pandas
    df_cv_metrics = pd.DataFrame(rows_results)
    # Imprimir tabla limpia en terminal
    print(df_cv_metrics.to_string(index=False))
    # Devolver resultado
    return df_cv_metrics

def get_oof_thresholds(audiences_top_pct,X_train_set, y_train_set, pipeline_final,n_splits=5):
    # 0. Mensaje del paso
    print("\n3.3. Obtención de umbrales para audiencias:")
    print("- Audiencias (Top %):", audiences_top_pct)
    # 1. Validación: deben ser porcentajes válidos y útiles para segmentar audiencia
    if any(not (0 < pct < 100) for pct in audiences_top_pct):
        raise ValueError("Los porcentajes de audiencia (Top %) deben estar estrictamente entre 0 y 100.")
    # 2. Definimos la tasa base train (proporción P)
    tasa_base_P_train = y_train_set.mean()
    # 3. Generamos las probabilidades OOF usando el mejor modelo con parámetros tuneados
    print("- Realizando OOF cross_val_predict basado en train set:\n")
    oof_preds = cross_val_predict(
        clone(pipeline_final), #Al usar clone, genera un pipeline limpio con ColumnTransformer y el modelo con hiperparámetros (sin datos train)
        X_train_set,
        y_train_set,
        cv=n_splits,
        method="predict_proba",
        n_jobs=-1
        )[:, 1] #Añadimos esto para obtener las probabilidades(propensity scores)
    # 4. Calculamos los umbrales congelados como percentiles: Top 5% (percentile 95%), Top 15% (percentile 85%), Top 30% (percentile 70%), etc.
    ## Inicializamos Resultados
    resultados_oof_limpios = []
    ## Calculos threshold y métricas para cada percentil (audiencia top)
    for top in audiences_top_pct:
        # Umbral
        tmp_th = np.percentile(oof_preds, (100 - top))
        ## Calculos
        preds_bin = (oof_preds >= tmp_th).astype(int) #Pasamos propensity scores a binaria (en base a ese umbral th)
        leads_fp = int(np.sum((y_train_set == 0) & (preds_bin == 1)))
        total_sel = int(np.sum(preds_bin == 1))
        prec = precision_score(y_train_set, preds_bin)
        #Guardamos
        resultados_oof_limpios.append({
                "Audiencia": f"Audiencia Top {top} %",
                "Set": f"Train (OOF):CV={n_splits}",
                "Umbral_Score": round(tmp_th, 4),
                "Precision": round(prec, 4),
                "Recall": round(recall_score(y_train_set, preds_bin), 4),
                "Lift (Precision / Prop_P)":round(prec / tasa_base_P_train, 2),
                "Leads_Retargeting_FP": leads_fp,
                "Total_Predicted_Positives": total_sel,
                "Perc_LeadsFP_Predicted_Positives": round((leads_fp / total_sel)*100, 2),
                "Perc_LeadsFP_Total_Users": round((leads_fp / len(y_train_set))*100, 2),
            })
    # 5. Convertir a dataframe resultados y dict de umbrales
    df_umbrales_oof = pd.DataFrame(resultados_oof_limpios)
    umbrales_dict = dict(zip(df_umbrales_oof["Audiencia"], df_umbrales_oof["Umbral_Score"]))
    # 6. Mostra umbrales
    print(df_umbrales_oof[["Audiencia", "Umbral_Score"]].to_string(index=False))
    # Devolver resultados
    return df_umbrales_oof, umbrales_dict

def save_train_results(pipeline_final, df_umbrales_oof, umbrales_dict, cv_train_metrics):
    # Guardamos resultados
    print("\n3.4. Guardando Resultados de Fase Entrenamiento (Training):")
    print(TRAIN_RESULTS_PATH)
    # Empaquetar en un dict train_results
    train_results = {
        "pipeline": pipeline_final,  # Incluye tu ColumnTransformer + LightGBM entrenados con train
        "train_cv_metrics":cv_train_metrics,
        "thresholds": umbrales_dict,
        "df_oof_results":df_umbrales_oof,
    }
    # Guardar resultados
    joblib.dump(train_results, TRAIN_RESULTS_PATH)

# Programa principal
def main():
    ##0. Mensaje nombre y version programa
    print(f"\n{PROGRAM_NAME} v {__version__}")
    ##1. Carga del processed dataset
    df_shoppers_processed = carga_processed_dataset()
    ## 2. Split en train y test con parámetros fijos y guardar tablas
    train_set, test_set = split_train_test(df_shoppers_processed)
    ##Separamos X e y para train set
    print("\nSeparamos Train Set en X e y ...")
    X_train_set = train_set.drop(columns=[TARGET])
    y_train_set = train_set[TARGET]
    ## 3. Fase de Training
    print("\n3. Fase de Entrenamiento (Training):")
    ## 3.1. Entrenamos el modelo con train e hiperparametros tuneados
    lightGBM_pipeline = train_model(X_train_set, y_train_set, transformer_base)
    ## 3.2. Calcular métricas CV
    cv_train_metrics = get_cv_metrics(X_train_set, y_train_set, lightGBM_pipeline, scoring)
    ## 3.3. Calcular umbrales 
    df_umbrales_oof, umbrales_dict = get_oof_thresholds(audiencias_top_pct, X_train_set, y_train_set, lightGBM_pipeline)
    ## 3.4. Guardar resultados de train
    save_train_results(lightGBM_pipeline, df_umbrales_oof, umbrales_dict, cv_train_metrics)

if __name__ == "__main__":
    main()