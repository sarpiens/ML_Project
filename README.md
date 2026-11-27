# Proyecto ML: Propensity Scoring en E-Commerce

## Índice

- [**Descripción**](#descripción)
- [**Contexto y Objetivos**](#contexto-y-objetivos)
- [**Modelos y Métricas**](#modelos-y-métricas)
- [**Resultados Obtenidos**](#resultados-obtenidos)
- [**Tecnologías y Prerrequisitos**](#tecnologías-y-prerrequisitos)
- [**Instrucciones de Ejecución**](#instrucciones-de-ejecución)

## Descripción

Proyecto individual del módulo de Machine Learning (ML) para el Curso de Data Science - The Bridge y Fundación Universia (2026-09).

Este proyecto usó datos del [**"Online Shoppers Purchasing Intention Dataset"**](https://doi.org/10.24432/C5F88Q) creado por C. Sakar y Yomi Kastro (2018), alojado en el "UCI Machine Learning Repository". El dataset original tiene una licencia [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/legalcode). Sobre los registros originales se aplicaron transformaciones de limpieza y preprocesamiento, particionado (*train/test*) y generación de variables derivadas. El código fuente de este proyecto se distribuye bajo la **Licencia MIT** (ver archivo `LICENSE`).

## Contexto y Objetivos

En el comercio electrónico, solo una pequeña parte de las visitas termina en compra. Invertir presupuesto de marketing de forma masiva sobre todo el tráfico resulta ineficiente. Por ello, se ha decidido centrar la temática en un problema de negocio e-commerce. El dataset seleccionado contiene información (18 columnas) referente a sesiones de diferentes usuarios de un comercio online (12330 filas) con una variable target (`Revenue`) que indica si la sesión terminó en venta o no. Lo que nos permite perseguir una serie de objetivos:

- **Objetivo técnico**: preparación de modelos ML de tipo supervisado aplicados a un problema de clasificación que nos permita clasificar a los usuarios en función de su comportamiento de sesión.

- **Objetivo de negocio**: utilizar los modelos ML preparados para obtener un propensity score (un predictor que estime la probabilidad de un usuario de comprar), que después podamos usar para identificar usuarios con un comportamiento de comprador pero que al final no compraron (Falsos Positivos como leads potenciales) para realizar futuras campañas de retargening con distintas audiencias (Top 5%, Top 15% y Top 30%).

## Modelos y Métricas

Inicialmente, se fijaron un total de 5 modelos candidatos supervisados de clasificación:

* `Regresión logística`

- `Random Forest` (Bagging)

* `XGBoost` (Boosting)

- `CatBoost` (Boosting)

* `LightGBM` (Boosting)

Por otro lado, se eligieron las siguientes métricas para la evaluación del modelo: 

* `Average Precision`: Evalúa cómo de bien concentra el modelo a los compradores potenciales en los scores altos sin dejarse engañar por los verdaderos negativos.

- `ROC-AUC`: Mide la capacidad global de ordenar correctamente a usuarios con alta intención frente a usuarios sin intención.

* `Brier Score`: Dirigida principalmente a la calibración de probabilidades. Penaliza probabilidades mal calibradas o excesivamente confiadas.

Asi mismo, también se aplicaron las siguientes métricas para la evaluación de umbrales:

- `Precision`: De todos los usuarios que superan el umbral, qué proporción sí compró en esa audiencia (`TP / Total de Positivos Predichos`). Nos ayuda a medir la calidad del patrón capturado.

* `Recall`: De todos los compradores reales del conjunto, qué proporción sí compró en esa audiencia (`TP / Total Positivos (TP + FN)`). Mide la Ganancia Acumulada (*Cumulative Gain*), o cobertura, es decir, cuánto comportamiento comprador estamos capturando.

- `Lift`: Mide cuántas veces supera el modelo a una selección aleatoria (`Precision / Tasa Base`, donde la proporción base de compradores en train y test es **0.155**).

* `Leads Retargeting (FP)`: Usuarios que superan el umbral pero no compraron. Representan nuestro público accionable (leads potenciales), es decir, el volumen de personas a las que impactar con diferentes estrategias de retargeting.

- `Total Predicted Positives`: Total de usuarios que superan el umbral en esa audiencia (`TP + FP`).

* `Perc. LeadsFP Predicted Positives`: El porcentaje de *leads* accionables (`FP`) respecto al total de Positivos Predichos en esa audiencia (`TP + FP`). Útil para ver qué parte de la audiencia target serían *leads* accionables.

- `Perc. LeadsFP Total Users`: El porcentaje de *leads* accionables (`FP`) respecto al total de usuarios del conjunto train. Útil para ver qué parte del total de usuarios del conjunto train serían *leads* accionables.

## Resultados Obtenidos

Tras comparar cinco algoritmos base y optimizar los mejores candidatos (`CatBoost` y `LightGBM`) mediante `GridSearchCV` (`CV=5`), el modelo ganador fue **`LightGBM` (Tuned)**.

### Evaluación del Modelo Final (Train CV vs. Test)

| Modelo | Versión | Set | Brier Score (↓) | Average Precision (↑) | ROC-AUC (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | *Baseline* | `Train CV=5` | 0.1160 | 0.3535 | 0.7728 |
| **LightGBM** | **Tuned** | `Train CV=5` | **0.1123** | **0.3833** | **0.7854** |
| **LightGBM** | **Tuned** | `Test` | **0.1149** | **0.3559** | **0.7698** |

El modelo final (`LightGBM Tuned`) generaliza de forma estable, más que duplicando la tasa base de azar (0.155) y manteniendo una calibración fiable sin señales de *overfitting* ni *underfitting* severo.

### Explicabilidad (Feature Importance y SHAP)

El análisis de *Feature Importance* y *SHAP* reveló que las variables relacionadas con exploración de productos (`ProductRelated`), las tasas de salida (`ExitRates`) y la estacionalidad mensual (`Month`) son los factores que más impulsan o penalizan al modelo final.

### Evaluación de Negocio (Train CV vs Test Set)

| Audiencia | Umbral_Score | Precision | Recall | Lift | Leads Retargeting (FP) | Total Predicted Positives | Perc. LeadsFP from Predicted Positives | Perc. LeadsFP from Total Users |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VIP (Top 5 %) Train CV=5** | 0.4215 | 0.5061 | 0.1638 | **3.27x** | 244 | 494 | 49.39 % | 2.47 % |
| **Estándar (Top 15 %) Train CV=5** | 0.2867 | 0.4000 | 0.3879 | **2.59x** | 888 | 1480 | 60.00 % | 9.00 % |
| **Amplia (Top 30 %) Train CV=5** | 0.2054 | 0.3258 | 0.6317 | **2.11x** | 1995 | 2959 | 67.42 % | 20.23 % |
| **VIP (Top 5 %) Test** | 0.4215 | 0.4130 | 0.1492 | **2.67x** | 81 | 138 | 58.70 % | 3.28 % |
| **Estándar (Top 15 %) Test** | 0.2867 | 0.4016 | 0.3953 | **2.59x** | 225 | 376 | 59.84 % | 9.12 % |
| **Amplia (Top 30 %) Test** | 0.2054 | 0.3058 | 0.6204 | **1.97x** | 538 | 775 | 69.42 % | 21.82 % |

En conjunto, al aplicar los umbrales para las distintas audiencias de negocio sobre el `Test Set` **las métricas mantienen su eficacia y estabilidad** cuando el modelo final se enfrenta a datos nuevos que no ha visto nunca, al compararlas con las métricas correspondientes obtenidas con el Train Set.

> Umbral_Score, corresponde al *propensity score* utilizado como umbral para definir las distintas audiencias de interés calculado sobre las predicciones del Train Set con validación-cruzada (CV).

## Tecnologías y Prerrequisitos

- **Lenguaje:** Python. Este análisis está desarrollado en **Python 3.12.10**. Los notebooks y scripts utilizan varias librerías integradas en la biblioteca estándar de Python.

- **Librerías Externas:** Para que el análisis funcione correctamente, es necesario instalar librerías de terceros (`pandas`, `numpy`, `scikit-learn`, `xgboost`,`lightgbm`, `catboost`, `scipy`,`shap`, entre otras). Se ha incluido un archivo `requirements.txt` con todas las librerías y versiones del entorno utilizado.

## Instrucciones de Ejecución

1. Clona el repositorio y acceder al directorio:
```bash
git clone https://github.com/sarpiens/ML_Project.git
cd ML_Project
```
2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecutar el flujo de preprocesado, entrenamiento y evaluación:
```bash
python src/data_processing.py
python src/training.py
python src/evaluation.py
```
4. Explorar los *notebooks* y la documentación en `docs`: la memoria (`Memoria.md`) y la presentación.
