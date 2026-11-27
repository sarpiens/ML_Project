#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: data_processing.py

Description:

Código para descargar y procesar los datos raw y guardar los datos procesados en la carpeta `data/processed`.

Proyecto individual curso Data Science - The Bridge y Fundación Universia (2026-09).

Author: Samuel PE
Date: 22 Noviembre 2026

Version: 1.0.0
"""

#Importar librerias 
import pandas as pd
import numpy as np
from pathlib import Path
from ucimlrepo import fetch_ucirepo

#Atributos del programa y constantes
PROGRAM_NAME = "data_processing.py"
__version__ = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
PROCESSED_PATH = PROJECT_ROOT / "data"/ "processed" / "online_shoppers_intention_processed.csv"

#Diccionario de meses columna Month en dataset(key): mes nombre completo (value)
month_dict = {'Feb':'February', 'Mar':'March', 'May':'May','June':'June', 'Jul':'July', 
                  'Aug':'August','Sep':'September','Oct':'October', 'Nov':'November', 'Dec':'December'}


#Variables seleccionadas después del EDA
TARGET = "Revenue"
features_finales = ["Administrative","Administrative_Duration", "Informational", "Informational_Duration",
                    "ProductRelated", "ProductRelated_Duration", "ProductRelated_Ratio", "ProductRelated_Duration_Ratio",
                    "SpecialDay","ExitRates","Month", "VisitorType","TrafficType"]
columnas_finales = features_finales + [TARGET]

#Funciones del programa
def fetch_raw_data():
    #0. Mensaje de paso actual
    print("\n1. Obtención y carga del dataset:")
    #1. Si el archivo existe localmente, que intente leerlo
    if RAW_PATH.exists():
        print(f"\nCargando dataset local desde {RAW_PATH}...")
        return pd.read_csv(RAW_PATH, sep=",")
    #2. Si no existe, que lo descarge con manejo de errores
    try:
        print("\nDescargando dataset desde UCI Machine Learning Repository...")
        raw_dataset = fetch_ucirepo(id=468)
        df_shoppers_raw = raw_dataset.data.original
        return df_shoppers_raw
    except Exception as e:
        error_text = f"\nNo se pudo descargar el dataset desde UCI Repository!\nRevisa tu conexión a internet o coloca manualmente el CSV en RAW_PATH.\nDetalle del error: {e}"
        raise RuntimeError(error_text) from e

def limpieza_inicial(df_shoppers, month_dict):
    #0. Mensaje paso actual
    print("\n2. Limpieza de variables de interés:")

    #Hacemos una copia
    df_shoppers_clean = df_shoppers.copy()
    
    #Variable Month
    print("- Month: homogenización a meses con nombre completo")
    ##Definimos un dict para reemplazar los valores a los nombres completos de los 10 meses presentes
    df_shoppers_clean["Month"] = df_shoppers_clean["Month"].replace(month_dict)

    #Variable Revenue (pasamos de buleano a binaria)
    print("- Revenue: conversión buleana a binaria (0,1)")
    df_shoppers_clean["Revenue"] = df_shoppers_clean["Revenue"].astype(int)

    return df_shoppers_clean

def derivacion_variables(df_shoppers):
    #0. Mensaje paso actual
    print("\n3. Derivación de variables:")

    #Hacemos un copia
    df_deriv = df_shoppers.copy()

    # 1. Calculamos los totales 
    ## Duración total de 3 categorías (total_duration) [esta forma de hacerlo propaga NAs, lo que queríamos]
    total_duration = (df_deriv["ProductRelated_Duration"] + df_deriv["Informational_Duration"] + df_deriv["Administrative_Duration"])
    ## Páginas totales de 3 categorías (total_pages) [esta forma de hacerlo propaga NAs, lo que queríamos]
    total_pages = (df_deriv["ProductRelated"] + df_deriv["Informational"] + df_deriv["Administrative"])

    # 1. Calculamos Proporción de Páginas de Producto (ProductRelated_Ratio)
    print("- ProductRelated_Ratio: ProductRelated / total_pages")
    df_deriv["ProductRelated_Ratio"] = np.where(total_pages.isna() | df_deriv["ProductRelated"].isna(), np.nan, #Controlamos NAs generados por datos faltantes
        np.where(total_pages != 0, df_deriv["ProductRelated"] / total_pages, 0) #Cuando la división sea entre cero devolvemos cero
        )

    # 2. Calculamos Duración asociada a Productos (ProductRelated_Duration_Ratio)
    print("- ProductRelated_Duration_Ratio: ProductRelated_Duration / total_duration")
    df_deriv["ProductRelated_Duration_Ratio"] = np.where(total_duration.isna() | df_deriv["ProductRelated_Duration"].isna(), np.nan, #Controlamos NAs generados por datos faltantes
        np.where(total_duration != 0, df_deriv["ProductRelated_Duration"] / total_duration, 0) #Cuando la división sea entre cero devolvemos cero
        )
    return df_deriv

def seleccion_columnas(df_shoppers, list_cols):
    #0. Mensaje paso actual
    print("\n4. Selección de variables finales:")
    print(list_cols)
    return df_shoppers[list_cols].copy()

def guardar_df_processed(df_shoppers):
    #0. Mensaje paso actual
    print("\n5. Guardamos Dataset procesado:")
    print(PROCESSED_PATH)
    #Guardamos el resultado en CSV
    df_shoppers.to_csv(PROCESSED_PATH, sep=",")

#Programa principal
def main():
    ##0. Mensaje nombre y version programa
    print(f"\n{PROGRAM_NAME} v {__version__}")
    ##1. Obtención y carga del dataset
    df_shoppers_raw = fetch_raw_data()
    ##2. Limpieza inicial
    df_shoppers_clean = limpieza_inicial(df_shoppers_raw, month_dict)
    ##3. Derivación de variables
    df_shoppers_clean_deriv = derivacion_variables(df_shoppers_clean)
    ##4. Filtrado de columnas finales
    df_shoppers_processed = seleccion_columnas(df_shoppers_clean_deriv, columnas_finales)
    ## 5. Guardar la tabla procesada
    guardar_df_processed(df_shoppers_processed)

if __name__ == "__main__":
    main()