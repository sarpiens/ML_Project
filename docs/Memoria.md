# Proyecto Machine Learning

## 1. Descripción

Proyecto individual del módulo de Machine Learning (ML) para el Curso de Data Science - The Bridge y Fundación Universia (2026-09). 

Este proyecto usó datos del [**"Online Shoppers Purchasing Intention Dataset"**](https://doi.org/10.24432/C5F88Q) creado por C. Sakar y Yomi Kastro (2018), alojado en el "UCI Machine Learning Repository". El dataset original tiene una licencia [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/legalcode). Sobre los registros originales se aplicaron transformaciones de limpieza y preprocesamiento, particionado (*train/test*) y generación de variables derivadas. El código fuente de este proyecto se distribuye bajo la **Licencia MIT** (ver archivo `LICENSE`).

## 2. Contexto y Objetivos

En el comercio electrónico, solo una pequeña parte de las visitas termina en compra. Invertir presupuesto de marketing de forma masiva sobre todo el tráfico resulta ineficiente. Por ello, se ha decidido centrar la temática en un problema de negocio e-commerce. El dataset seleccionado contiene información (18 columnas) referente a sesiones de diferentes usuarios de un comercio online (12330 filas) con una variable target (`Revenue`) que indica si la sesión terminó en venta o no. Lo que nos permite perseguir una serie de objetivos:

- **Objetivo técnico**: preparación de modelos ML de tipo supervisado aplicados a un problema de clasificación que nos permita clasificar a los usuarios en función de su comportamiento de sesión.

- **Objetivo de negocio**: utilizar los modelos ML preparados para obtener un propensity score (un predictor que estime la probabilidad de un usuario de comprar), que después podamos usar para identificar usuarios con un comportamiento de comprador pero que al final no compraron (Falsos Positivos como leads potenciales) para realizar futuras campañas de retargening con distintas audiencias (Top 5%, Top 15% y Top 30%).

## 3. Etapas del Proyecto

### 3.1. Exploración, Limpieza Inicial y Derivación de Variables

Los resultados detallados de esta etapa pueden consultarse en el notebook `1.Preparacion_Dataframe.ipynb`. Así mismo, el archivo `src/data_processing.py` contiene el código para descargar y procesar los datos raw y guardar los datos procesados en la carpeta `data/processed`, con la selección final de variables de interés.

> **Nota:** El archivo `online_shoppers_intention_clean.csv` resultado de la limpieza y derivación en fase exploratoria del notebook se usó como versión intermedia y está localizado en la carpeta `data/interim` donde se han guardado estos y otros archivos intermedios.

#### 3.1.1. Exploración del Dataset y sus Variables

El dataset original tiene un total de 12330 entradas que según su descripción cada sesión corresponde a un único usuario en un periodo de un año para evitar tendencias a campañas específicas, días especiales, perfiles de usuarios, o periodos. 

Hay un total de 18 columnas:

- Las primeras 6 (Columnas 0-5) representan el número de página visitadas en la sesión para cada categoría de página (`Administrative`, `Informational` y `ProductRelated`) y el total de tiempo empleado en cada categoría (`Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`). Los números de páginas son variables númericas discretas, mientras que los tiempos de duración son variables númericas continuas.

* Las siguientes 3 (Columnas 6-8) son métricas agregadas de "Google Analytics" que representan el valor promedio de los indicadores históricos de todas las páginas que el usuario visitó durante su sesión. Las 3 son variables numéricas continuas:

    - `BounceRates`: promedio de las tasas de rebote de todas las URLs visitadas en la sesión (parece estar en proporciones). Las tasas de rebote en Google Analytics, reportan el porcentaje de visitantes que acceden al sitio desde esa página y luego lo abandonan (rebotan) sin generar ninguna otra solicitud al servidor de análisis durante esa sesión.     

    * `ExitRates`: promedio de las tasas de salida de todas las URLs visitadas en la sesión (parece estar en proporciones). Las tasas de salida en Google Analytics, reportan el porcentaje de todas las visitas a dicha página que constituyeron la última visita de la sesión.
           
    - `PageValues`: promedio del valor monetario de todas las URLs visitadas en la sesión. El valor de las páginas en Google Analytics, reporta el valor monetario promedio de una página web que un usuario ha visitado antes de completar una transacción de comercio electrónico. Para nuestro objetivo, será conveniente eliminarla antes del modelado. Por cómo está construida, podría introducir un cierto *data leakage* y contaminar el *propensity score* que queremos generar más adelante. Las páginas finales del proceso de compra muy probablemente dispararán estos valores, generando una correlación muy fuerte con la variable target (`Revenue`). Esto no nos interesa, ya que nos gustaría que el modelo identificara usuarios que mostraron comportamientos similares a los compradores para futuras campañas de retargeting de los no compradores.
    
- `SpecialDay` (Columna 9): indica la proximidad de la fecha de visita al sitio web a un día especial, en el que es más probable que las sesiones concluyan en transacción. Este valor se determina teniendo en cuenta la dinámica del comercio electrónico, como el intervalo de tiempo entre la fecha del pedido y la fecha de entrega. Por ejemplo, para San Valentín, toma un valor distinto de cero entre el 2 y 12 de Febrero, es cero antes y después de esta fecha (a menos que se acerque a otra fecha especial) y su valor máximo de 1 se alcanza el 8 de Febrero. Por lo que es una variable numérica continua.

* `Month` (Columna 10): indica el mes en la que la sesión tuvo lugar. Solo hay representados 10 de los 12 meses posibles, por lo que es una limitación a tener en cuenta más adelante para tratar esta variable categórica. Además, los nombres no están estandarizados algunos aparecen con abreviaturas y otros con el nombre completo.

- Columnas Codificadas (Columnas 11-14), tenemos `OperatingSystems`, `Browser`, `Region`, `TrafficType`, al estar codificadas carecen de significado interpretable, ya que no sabemos lo que significa cada categoría. Probablemente sea recomendable eliminarlas más adelante antes del modelado, ya que conservar variables que no podemos explicar puede dificultar la interpretación. De cara al análisis exploratorio se decidió conservarlas y postergar esta decisión.

* `VisitorType` (Columna 15): indica el tipo de visitante como 'Returning_Visitor', 'New_Visitor', o 'Other'. Por lo que se trata de una variable categórica.

- `Weekend` (Columna 16): indica si la visita tuvo lugar en fin de semana. Además, está como un buleano así que nos interesará pasarlo a binario.

* `Revenue` (Columna 17): indica si la visita acabo en una compra o no. Esta es la **variable target de interés**. Además, también está como un buleano así que nos interesará pasarlo a binario.

#### 3.1.2. Limpieza Inicial de Variables

- Variable `Month`: estandarización de valores a los nombres completos de los meses en Inglés.

* Variable `Weekend`: conversión de variable buleana a binaria (0,1).

- Variable `Revenue`: conversión de variable buleana a binaria (0,1).

#### 3.1.3. Derivación de Variables

También se generaron algunas variables de interés centradas en productos:

- **Proporción de Páginas de Producto** (`ProductRelated_Ratio`):

    * **Razonamiento**: para un problema de negocio nos puede interesar saber el total de las páginas relacionadas con productos (`ProductRelated`) con respecto al global de páginas visitadas. Así que tiene sentido ver cual es el ratio de páginas de producto con respecto al total de páginas.

    - **Cálculo**: `ProductRelated / total_pages` (el resultado deberá ser 0 cuando total_pages sea 0, para evitar dividir por cero, pero también considerar NaNs por valores faltantes), donde:

        * total_pages = `ProductRelated + Informational + Administrative`

* **Duración asociada a Productos** (`ProductRelated_Duration_Ratio`):

    - **Razonamiento**: para un problema de negocio también nos interesa saber el tiempo que pasa el usuario en páginas relacionadas con productos (`ProductRelated_Duration`) con respecto al tiempo global de la sesión. Así que tiene sentido ver cual es el ratio de tiempo que pasa en páginas de productos con respecto al tiempo total.

    * **Cálculo**: `ProductRelated_Duration / total_duration` (el resultado deberá ser 0 cuando total_duration sea 0, para evitar dividir por cero, pero también considerar NaNs por valores faltantes), donde:

        - total_duration = `ProductRelated_Duration + Informational_Duration + Administrative_Duration`

- **Duración por Página de Producto** (`Duration_Per_Product_Page`):

    * **Razonamiento**: para un problema de negocio también nos puede interesar fijarnos en cuanto tiempo a pasado el usuario por cada página asociada a productos.

    - **Cálculo**: `ProductRelated_Duration / ProductRelated` (el resultado deberá ser 0 cuando ProductRelated sea 0, para evitar dividir por cero, pero también considerar NaNs por valores faltantes).

En este caso, si que podemos realizar la derivación de variables antes del split train-test ya que no tenemos valores faltantes (NaN) en ninguna de las variables asociadas originales. Tal y como se ha preparado el código si huviera algún NaN en las variables originales se generaría un NaN que posteriormente se podría imputar, lo que aportaría mayor control sobre que datos se imputan.

### 3.2. Análisis Exploratorio del Train Set

Los resultados detallados de esta etapa de análisis exploratorio de datos (EDA) centrado en el Train Set y sus gráficos asociados pueden consultarse en el notebook `2.EDA_TrainSet.ipynb` y en la presentación asociada. Así mismo, se realizó un split provisional entre train y test sets aplicando:

* Proporción 80-20 para train-test (`test_size=0.2`).

- Fijando el random state (`random_state=42`) para reproducivilidad.

* Estratificando teniendo en cuenta la variable target, ya que se vió que estaba desbalanceada con un 15.5% de compradores (`stratify=df_shoppers["Revenue"]`).

> **Nota:** Estos dos sets fueron usados como versiones intermedias para realizar el EDA inicial sobre el train set antes de la selección de variables. Están en el archivo `df_shoppers_split.joblib` localizado en la carpeta `data/interim` donde se han guardado estos y otros archivos intermedios. También se generó un archivo `train_set_nums_prep.csv` intermedio con las transformaciones numéricas que fueron usadas en el EDA.

#### 3.2.1. Análisis Univariante

La descripción de estadísticos, gráficos y comentarios detallados del análisis univariante pueden consultarse en los apartados 2.1 y 2.2 del notebook correspondiente. Se generaron gráficos de barras (variables categóricas), así como boxplots e histogramas (variables numéricas). Cabe destacar que:

- Todas las categorías posibles del dataset original han quedado representadas en el Train Set, garantizando que no habrá categorías faltantes durante la fase de entrenamiento.

* Al ver las distribuciones de las variables numéricas:
    - Encontramos ciertas variables (`Administrative`, `Informational`, `ProductRelated`, `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `PageValues` y `Duration_Per_Product_Page`) relacionadas con conteos, duraciones y valores monetarios que presentan grandes picos a la izquierda cerca de cero y colas a la derecha. Por ello, se decidió aplicar una **transformación logarítmica `np.log1p`** para simetrizar y comprimir la escala.
    - También tenemos los ratios de proporción de producto (`ProductRelated_Ratio` y `ProductRelated_Duration_Ratio`) que muestran grandes picos a la derecha cercanos a 1 y colas a la izquierda. Por otro lado, `SpecialDay` tiene un gran pico en torno a cero y escalones discretos hasta 1. Además, `BounceRates` y `ExitRates` presentan una bimodalidad con un pico principal cercano a cero y un segundo pico menor en torno a 0.2. En todos estos casos, no se aplicó una logarítmica, pues ya están dentro de una escala acotada 0-1 que es de por sí comprimida y no corregiría ni la asimetria ni la bimodalidad.
    - Finalmente, a todas estas variables numericas se decidió aplicar el mismo escalado **`StandardScaler`** para centrar sus distribuciones, y garantizar que sean comparables entre sí de cara a modelos sensibles a la escala.

#### 3.2.2. Análisis Bivariante dirigido al Target (Revenue)

Los gráficos y comentarios detallados del análisis dirigido al target pueden consultarse en el apartado 2.3 del notebook correspondiente. Se generaron gráficos de barras (variables categóricas), así como análisis de asociación y correlación dirigidos al target (aprovechando que era binaria) contra las variables categóricas (Test V Cramer) y numéricas (Spearman: Train Set Sin Transformar y Pearson: Train Set Transformado). Cabe destacar que:

- `Revenue==0` se mantiene como el grupo mayoritario en la gran mayoría de categorías de las distintas variables. Aunque en ciertos casos la distancia relativa entre Revenues se acortan, la tendencia general de desbalanceo se conserva.

* Con respecto a los resultados de asociación del Test V Cramer:
    - Vemos que las variables categóricas que presentan mayor asociación con la variable target están en un rango V Cramer 0.1-0.2, `TrafficType`, `Month` y `VisitorType`. Estas aportarían una señal de débil a moderada (0.1-0.3), así que serían interesantes de incorporar a los distintos modelos más adelante.
    - Mientras que el resto de variables presentan una asociación muy débil en un rango entre 0.02-0.08. Así que podríamos plantearnos eliminarlas y no incorporarlas en los distintos modelos más adelante para reducir la dimensionalidad.

- Con respecto a los resultados de correlación (Spearman y Pearson):
    - `PageValues` es la variable con mayor correlación con la variable target como ya predijimos (Pearson=0.64 y Spearman=0.63). Para nuestro objetivo, será conveniente eliminarla antes del modelado, ya que podría introducir un cierto *data leakage* y contaminar el *propensity score* que queremos generar más adelante. Esto no nos interesa, ya que nos gustaría que el modelo identificara usuarios que mostraron comportamientos similares a los compradores para futuras campañas de retargeting de los no compradores.
    - `ProductRelated_Duration_Ratio` muestra una correlación muy baja con el target cercana a cero (aunque curiosamente cambia el signo, Pearson=0.07 y Spearman=-0.06).
    - Mientras que el resto de variables presentan correlaciones similares en ambos casos. Encontramos correlaciones positivas con el target, en rango 0.22-0.1(`ProductRelated`,`ProductRelated_Duration`, `Administrative`, `Administrative_Duration`, `Duration_Per_Product_Page`, `Informational`, `Informational_Duration`). También correlaciones negativas con el target en un rango -0.05 a -0.26 (`ProductRelated_Ratio` y `SpecialDay`, `BounceRates` y `ExitRates`), siendo `BounceRates` (Pearson=-0.15 y Spearman=-0.16) y `ExitRates` (Pearson=-0.21 y Spearman=-0.26) las que presenten mayor correlación negativa.

#### 3.2.3. Análisis Multivariante entre Features

Los gráficos y comentarios detallados del análisis entre features pueden consultarse en el apartado 2.4 del notebook correspondiente. Estos análisis se centraron en explorar para las distintas features las asociaciones y correlaciones entre variables categóricas (Test V Cramer), numéricas (Spearman: Train Set Sin Transformar y Pearson: Train Set Transformado) y numéricas-categóricas (Asociación eta2 junto con Test Kruskal-Wallis). Cabe destacar que:

* Con respecto a los resultados de asociación del Test V Cramer:
    - Hay una colinealidad fuerte entre `OperatingSystems` y `Browser` (V = 0.6), así como asociaciones moderadas con `VisitorType` (V = 0.46 y V = 0.5, respectivamente) y débiles con `TrafficType` (V en torno a 0.2). También vemos que `TrafficType` tiene asociaciones débiles con `Weekend` (V = 0.15), y `VisitorType` con `Region` (V = 0.18). De manera que parte de esa información técnica, temporal y geográfica queda reflejada también en las tres variables categóricas de interés antes mencionadas (`VisitorType`, `TrafficType`, y `Month`).
    - Las tres variables categóricas de interés muestran asociaciones débiles o moderadas entre sí (V entre 0.15 y 0.39), lo que indicaría que no son redundantes aportando información complementaria sin introducir problemas de multicolinealidad.

- Con respecto a los resultados de correlación (Spearman y Pearson):
    - Vemos que la estructura general de correlaciones se mantiene antes y después de aplicar las transformaciones, aunque al aplicar estas transformaciones los valores de correlación aumentan en varios casos.
    - Al comparar las variables referentes a las 3 categorías de páginas (`Administrative`,`Informational`, y `ProductRelated`) entre su número páginas y su respectiva duración hay una alta correlación (R = 0.86-0.93), como cabría esperar, pero al proporcionar dimensiones distintas, vamos a conservarlas.
    - En el caso de `BounceRates` y `ExitRates`, están muy correlacionadas (R = 0.91) y ambas miden abandono, así que será recomendable eliminar `BounceRates` y conservar `ExitRates` que será más informativa.
    - También vemos que la variable derivada `Duration_Per_Product_Page` muestra una correlación fuerte con `ProductRelated_Duration` (R = 0.85), como miden una dimensión temporal similar y por mantener una estructura homogénea con las otras variables `Administrative_Duration` y `Informational_Duration`, será recomendable eliminar `Duration_Per_Product_Page` y conservar `ProductRelated_Duration`.
    - Finalmente, el resto de correlaciones serían moderadas o bajas (|R| < 0.8), por lo que en principio no deberían ser problemáticas entre sí.

* Con respecto a los resultados de asociación del coeficiente eta2:
    - Vemos algunas asociaciones fuertes (> 0.14), entre `Month` y `SpecialDay` (eta2 = 0.2812), lo cual tiene sentido ya que las fechas especiales están en meses concretos.
    - También moderadas (0.06-0.14), en especial en `TrafficType` (`ProductRelated_Ratio` con eta2 = 0.0673, `ExitRates` con eta2 = 0.1169 y `BounceRates` con eta2 = 0.1024) y `VisitorType` (`ExitRates` con eta2 = 0.078 y `BounceRates` con eta2 = 0.0929). Lo que indicaría que el patrón de navegación y abandono varía según el canal y el tipo de usuario.
    - Mientras que el resto presentan asociaciones pequeñas (0.01-0.06) o nulas (< 0.01), indicando que aportan información mayoritariamente complementaria sin redundancia importante.

#### 3.2.4. Selección de Variables

En base a los resultados del EDA centrado en el Train Set, se decidió no incluir las siguientes variables en los modelos de machine learning:

- Las variables `Region`, `Weekend`, `Browser`, y `OperatingSystems` presentan una asociación muy débil con la variable target en un rango V Cramer 0.02-0.08, así que no se incluyeron en los modelos para reducir ruido y dimensionalidad.

* `PageValues` es la variable con mayor correlación con la variable target. Para nuestro objetivo, será conveniente eliminarla antes del modelado, ya que podría introducir un cierto *data leakage* y contaminar el *propensity score* que queremos generar más adelante. Esto no nos interesa, ya que nos gustaría que el modelo identificara usuarios que mostraron comportamientos similares a los compradores para futuras campañas de retargeting de los no compradores (leads potenciales).

- En el caso de `BounceRates` y `ExitRates`, están muy correlacionadas (R = 0.91) y ambas miden abandono, así que vamos a eliminar `BounceRates` y quedarnos con `ExitRates` que será más informativa.

* También vemos que la variable derivada `Duration_Per_Product_Page` muestra una correlación fuerte con `ProductRelated_Duration` (R = 0.85). Como miden una dimensión temporal similar y por mantener una estructura homogénea con las otras variables, vamos a eliminar `Duration_Per_Product_Page` y conservar `ProductRelated_Duration`.

De manera que nuestras variables finales fueron:

- **Target**: `Revenue` (Binaria, Categórica).

* **Features**:
    - Categóricas: `Month`, `VisitorType` y `TrafficType`.
    * Numéricas: `ProductRelated`,`ProductRelated_Duration`, `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated_Ratio`, `ProductRelated_Duration_Ratio`, `SpecialDay` y `ExitRates`.

### 3.3. Entrenamiento y Evaluación de Modelos ML

Los resultados detallados de estas etapas de entrenamiento (Train) y evaluación (Test) pueden consultarse en el notebook `3.Entrenamiento_Evaluacion.ipynb` y en la presentación asociada.

> **Nota:** A partir de este punto se trabajó directamente con los archivos finales Train y Test (con la selección final de variables), generados por `src/training.py` y localizados en `data/train` y `data/test`, respectivamente. Los resultados principales de las fases de entrenamiento y evaluación generados por `src/training.py` y `src/evaluation.py`, así como los modelos ajustados por `GridSearchCV`, se encuentran localizados en la carpeta `models`.

#### 3.3.1. Entendiendo el ColumnTransformer

Se ha preparado un `ColumnTransformer` con todas las transformaciones necesarias que ha quedado almacenado en `src/training.py`, incluyendo:

* **log_scaler**: que incorpora un `SimpleImputer` con `strategy='median'` (para este dataset no sería necesario, pero lo hemos incluido por robustez de cara a futuros valores NaN), seguido por una transformación logarítmica `log1p` y un `StandarScaler`. Que se aplica a las columnas `Administrative`, `Informational`, `ProductRelated`, `Administrative_Duration`, `Informational_Duration` y `ProductRelated_Duration`. 

- **num_scaler**: que incorpora un `SimpleImputer` con `strategy='median'` (para este dataset no sería necesario, pero lo hemos incluido por robustez de cara a futuros valores NaN), seguido por un `StandarScaler`. Que se aplica a las columnas `ProductRelated_Ratio`, `ProductRelated_Duration_Ratio`, `SpecialDay` y `ExitRates`.

* **cat_std**: que utiliza un `OneHotEncoder` con las opciones `sparse_output=False` y `handle_unknown="ignore"`. Que hemos usado para las columnas `Month` y `VisitorType`.

- **cat_TrafficType**: que se aplica unicamente a la columna `TrafficType`. Primero, se aplica `asarray` para convertir los valores numéricos a str y así asegurarnos que si en un futuro entraran datos NaN (float) no alteraran los nombres de columnas pues esta variable es de tipo int de partida. Después, se ha aplicado un `OneHotEncoder` con las opciones `sparse_output=False`, `handle_unknown="ignore"` y `min_frequency=0.01`. Esta última agrupará los canales conocidos que son poco frecuentes (< 1%) en una columna `TrafficType_infrequent_sklearn` para así reducir la cardinalidad de esta variable que tenía 20 grupos.

Después de aplicar el ColumnTransformer, pasamos a tener 35 columnas de features:

- 11 columnas numéricas transformadas.

* 24 columnas categóricas pasadas a binarias (10 meses, 3 grupos de visitantes, y 11 tipos tráfico > 1% con una columna extra que agrupa canales minoritarios conocidos).

#### 3.3.2. Fase de Entrenamiento (Training) de Modelos ML

##### 3.3.2.1. Modelos Candidatos

Inicialmente, se fijaron un total de 5 modelos candidatos supervisados de clasificación:

* Regresión logística

- Random Forest (Bagging)

* XGBoost (Boosting)

- CatBoost (Boosting)

* LightGBM (Boosting)

##### 3.3.2.2. Selección de Métricas

Se eligieron las siguientes métricas: 

* `Average Precision`: Al tener un dataset desbalanceado con menos compradores, esta métrica nos permite evaluar cómo de bien concentra el modelo a los compradores potenciales en los scores altos sin dejarse engañar por los verdaderos negativos. Es decir, el rendimiento sobre la clase minoritaria. Además, resume la curva Precision-Recall mediante la suma ponderada de la precisión en cada umbral.

- `ROC-AUC`: Mide la probabilidad de que el modelo asigne un *propensity score* más alto a un comprador aleatorio que a un no comprador aleatorio. Estándar para validar la capacidad global de ordenamiento (ranking), mide la capacidad global de ordenar correctamente a usuarios con alta intención frente a usuarios sin intención.

* `Brier Score`: Dirigida principalmente a la calibración de probabilidades. Mide el error cuadrático medio directo de la probabilidad y es más estable ante outliers o probabilidades bajas. Penaliza probabilidades mal calibradas o excesivamente confiadas (que distorsionen las probabilidades hacia arriba). Importante si vamos a usar el número de probabilidad como *propensity score*.

##### 3.3.2.3. Entrenamiento de Modelos Baseline

En esta fase se compararon los 5 modelos candidatos en igualdad de condiciones con sus parámetros por defecto para establecer una línea base. Para ello, se utilizó el Train Set mediante cross-validation (CV=5). Los parámetros exactos pueden consultarse en el apartado 3.3.1 del notebook correspondiente. Esto dio lugar a los siguientes resultados:

| Modelo | Avg_Precision (CV=5; ↑ mayor es mejor) | ROC_AUC (CV=5; ↑ mayor es mejor) | Brier_Score (CV=5; ↓ menor es mejor) |
| :--- | :---: | :---: | :---: |
| **CatBoost** | 0.3539 ± 0.0118 | 0.7742 ± 0.0048 | 0.1157 ± 0.0012 |
| **LightGBM** | 0.3535 ± 0.0087 | 0.7728 ± 0.0060 | 0.1160 ± 0.0010 |
| **Random Forest** | 0.3424 ± 0.0173 | 0.7588 ± 0.0088 | 0.1172 ± 0.0019 |
| **Logistic Regression** | 0.3411 ± 0.0049 | 0.7587 ± 0.0066 | 0.1163 ± 0.0005 |
| **XGBoost** | 0.3395 ± 0.0190 | 0.7588 ± 0.0110 | 0.1226 ± 0.0035 |

En base a estos resultados, se decidió continuar con `CatBoost` y `LightGBM` para la siguiente fase de optimización de hiperparámetros ya que presentaban las mejores métricas.

##### 3.3.2.4. Ajuste de Hiperparámetros para Modelos Seleccionados

En esta fase se realizó un ajuste de hiperparámetros para los dos modelos seleccionados (`CatBoost` y `LightGBM`) mediante la estrategia `GridSearchCV`. Se utilizó el Train Set mediante cross-validation (CV=5). Los parámetros exactos pueden consultarse en el apartado 3.3.2 del notebook correspondiente. Cabe mencionar que los hiperparámetros fueron seleccionados para cada modelo teniendo en mente su futuro uso como *propensity score*, por lo que no se incluyeron parámetros de auto-balanceo para evitar inflar artificialmente las probabilidades de los modelos. Esto dio lugar a los siguientes resultados:

| Modelo | Versión | Brier Score (CV=5; ↓ menor es mejor) | Average Precision (CV=5; ↑ mayor es mejor) | ROC-AUC (CV=5; ↑ mayor es mejor) |
| :--- | :--- | :---: | :---: | :---: |
| **CatBoost** | *Baseline* | 0.1157 ± 0.0012 | 0.3539 ± 0.0118 | 0.7742 ± 0.0048 |
| **CatBoost** | *Tuned* | 0.1131 ± 0.0008 | 0.3737 ± 0.0126 | 0.7807 ± 0.0031 |
| **LightGBM** | *Baseline* | 0.1160 ± 0.0010 | 0.3535 ± 0.0087 | 0.7728 ± 0.0060 |
| **LightGBM** | ***Tuned* (Ganador)** | 0.1123 ± 0.0008 | 0.3833 ± 0.0190 | 0.7854 ± 0.0042 |

En base a estos resultados, se decidió continuar con el modelo `LightGBM` optimizado para la siguientes fases ya que presentaba las mejores métricas.

#### 3.3.3. Fase de Evaluación (Testing) del Modelo Final LightGBM Tuned

##### 3.3.3.1. Evaluación Global del Modelo Final (Train CV vs Test)

En esta fase se realizó la evaluación global del modelo final obteniendo métricas basadas en el Test Set, un conjunto de datos nunca visto por el modelo para poder comparar con los resultados previos de la fase de entrenamiento. Los resultados detallados pueden consultarse en el apartado 3.4.1 del notebook correspondiente. Esto dio lugar a los siguientes resultados:

| Modelo | Versión | Set | Brier Score (↓ menor es mejor) | Average Precision (↑ mayor es mejor) | ROC-AUC (↑ mayor es mejor) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | *Baseline* | Train CV=5 |0.1160 | 0.3535 | 0.7728 |
| **LightGBM** | **Tuned** | Train CV=5 |0.1123 | 0.3833 | 0.7854 |
| **LightGBM** | **Tuned** | Test |0.1149 | 0.3559 | 0.7698 |

Para el modelo final observamos que sus:
- `Brier Scores` son inferiores al `Brier base = 0.155 * (1-0.155) = 0.131`, mostrando una calibración adecuada con un modelo que reduce entre un 12.3% y 14.3% (Brier Skill Score) el error cuadrático de probabilidad frente al azar.
* `Average Precision` son superiores a su línea base de azar (0.155), más que duplicándola, manteniendo una capacidad sólida para concentrar compradores en los primeros puestos del ranking.
- `ROC-AUC` presentan valores ampliamente superiores al 0.7, validando su capacidad global de ordenamiento (ranking) entre usuarios con alta y baja intención de compra.

En conjunto, vemos que el rendimiento del modelo optimizado (`Tuned`) se mantiene estable con pequeñas variaciones al comparar las validaciones cruzadas previas (`Train CV`) y el conjunto final de test (`Test`), lo que indicaría ausencia de sobreajuste significativo, una buena generalización y alta estabilidad del modelo final.

##### 3.3.3.2. Explicabilidad (Feature Importance y SHAP)

Aśi mismo también se evaluó la explicabilidad del modelo final, explorando sus Feature Importance (de forma individual y por grupos de features) y mediante SHAP Summary (Beeswarm) plots. Los resultados detallados y gráficos asociados pueden consultarse en el apartado 3.4.3 del notebook correspondiente.

a) Feature Importance Individual (si nos centramos en las 15 primeras para Gain):

- `ExitRates`, es **la mayoritaria tanto en Gain (30.11%) como Split (16.18%)**.

* Las del `Grupo ProductRelated` suelen estar entre **las que más porcentaje explican** (ProductRelated_Duration, ProductRelated, ProductRelated_Ratio, ProductRelated_Duration_Ratio), siendo `ProductRelated_Duration` la que **más explica dentro de este grupo tanto en Gain (17.73%) como Split (13.17%).**

- Encontramos algunas derivadas de la variable `Month` (Month_November, Month_March, Month_October, Month_May), siendo `Month_November` la que **más explica dentro de este grupo tanto en Gain (6.01%) como Split (3.52%).** También otras temporales como `SpecialDay`, con valores menores (Gain = 1.32% y Split = 2.00%).

* Así mismo, encontramos variables relacionadas con otras clasificaciones de páginas web. Las del `Grupo Administrative` (Administrative_Duration y Administrative), siendo `Administrative_Duration` la que **más explica dentro de este grupo tanto en Gain (	5.04%) como Split (7.58%).** También del `Grupo Informational`, concretamente `Informational_Duration` con valores menores (Gain = 1.83% y Split = 3.77%).

- También vemos algunas relacinadas con el `VisitorType`, como `VisitorType_Returning_Visitor` (Gain = 2.76% y Split = 1.20%) y `VisitorType_New_Visitor` (Gain = 1.54% y Split = 0.94%).

b) Feature Importance por Grupos:

Al agrupar features por grupos de variables, los primeros 5 grupos se mantienen en el mismo orden tanto en Gain como Split, siendo el **`Grupo ProductRelated` el mayor** (Gain = 35.4% y Split = 39.7%), seguida por `ExitRates` (Gain = 30.1% y Split = 16.2%), `Month` (Gain = 13.1% y Split = 13.9%), `Administrative` (Gain = 7.9% y Split = 11.9%) y `TrafficType` (Gain = 5.5% y Split = 9.0%). Vemos que el `Grupo ProductRelated` y `ExitRates` **acumulan la mayor parte del impacto preditivo (65.5%) y frecuencia de cortes (55.9%)**. Mientras que el resto de grupos muestran porcentajes menores.

c) SHAP Summary (Beeswarm) plots:

- `ExitRates` tiene una relación inversa con la propensión de compra, valores menores implican mayor propensión de compra y sigue siendo la feature más importante.

* Del `Grupo ProductRelated`, se aprecian distintos patrones, `ProductRelated_Duration` tiene una relación positiva (mayor duración de la sesión en páginas relacionadas con productos mayor compra), mientras que `ProductRelated_Ratio` es negativa (cuando vemos otras categorías de páginas hay mayor compra).

- Del `Grupo Administrative`, `Administrative_Duration` tiene una relacion positiva (mayor duración de la sesión en páginas relacionadas con administración mayor compra), entre otros.

* Con respecto a la variables `Month` y `TrafficType`, vemos que hay meses con relaciones positivas y negativas para el modelo (Noviembre suele traducirse en mayor propensión de compra, mientras que Marzo y Mayo suele ser menor), y también los tipos de tráfico (Tipo 2 se traduce en mayor compra mientras los tipos 3 y 13 en menor compra).

### 3.4. Propensity Score y Retargeting

Los resultados detallados de estas etapas dirigidas al problema de negocio pueden consultarse en el notebook `3.Entrenamiento_Evaluacion.ipynb` y en la presentación asociada.

#### 3.4.1. Problema de Negocio

En este caso, vamos a definir **3 cortes de interés por capacidad de campaña**. Simularemos que el departamento de marketing tiene presupuesto para impactar distintos porcentajes de los usuarios con mayor *propensity score* según nuestro modelo buscando leads potenciales:

- **Audiencia VIP (Top 5%)**: Usuarios con mayores propensity scores priorizando calidad. Sus Falsos Positivos (FP, usuarios que no compraron pero navegaron de forma similar a un comprador) serán interesantes para campañas de alto coste o incentivos como cupones descuento.

* **Audiencia Estándar (Top 15%)**: Usuarios que presentan un equilibrio entre volumen y calidad. Sus FP serán interesantes para hacer un retargeting con campañas de coste medio, como anuncios dirigidos.

- **Audiencia Amplia (Top 30%)**: Un rango mayor de los usuarios priorizando volumen. Sus FP serán interesantes para retargeting mediante canales económicos de coste marginal cero,en especial canales propios, como email o notificaciones push.

#### 3.4.2. Ajuste del Umbral de Negocio (Train Set)

El primer paso consistió en la obtención de umbrales de corte para las distintas audiencias de interés. Para ello, se usaron predicciones Out-of-Fold (OOF) sobre `X_train_set` mediante validación cruzada (CV=5), asegurando que cada usuario reciba una probabilidad (que se uso como **propensity score**) predicha por un modelo que no lo vio al entrenar. A partir de esta distribución se obtuvieron los umbrales que marcan el corte del Top 5%, Top 15% y Top 30% de usuarios con mayor *propensity score*. Los resultados detallados pueden consultarse en el apartado 3.3.3 del notebook correspondiente.

Se calcularon las siguientes métricas de interés:

- `Precision`: De todos los usuarios que superan el umbral, qué proporción sí compró en esa audiencia (`TP / Total de Positivos Predichos`). Nos ayuda a medir la calidad del patrón capturado. Por ejemplo, una Precision de **0.3** indicaría que el **30%** de los usuarios seleccionados de la audiencia sí compraron.

* `Recall`: De todos los compradores reales del conjunto, qué proporción sí compró en esa audiencia (`TP / Total Positivos (TP + FN)`). Mide la Ganancia Acumulada (*Cumulative Gain*), o cobertura, es decir, cuánto comportamiento comprador estamos capturando. Por ejemplo, con un Recall de **0.3** lograríamos capturar el **30%** de todo el comportamiento comprador del train set.

- `Lift`: Mide cuántas veces supera el modelo a una selección aleatoria (`Precision / Tasa Base`, donde la proporción base de compradores en train es **0.155**). Por ejemplo, si alcanzamos un lift **3X**, significa que para esa audiencia concentramos el triple de intención de compra que el puro azar.

* `Leads_Retargeting_FP`: Usuarios que superan el umbral pero no compraron. Representan nuestro público accionable (leads potenciales), es decir, el volumen de personas a las que impactar con diferentes estrategias de retargeting.

- `Total_Predicted_Positives`: Total de usuarios que superan el umbral en esa audiencia (`TP + FP`).

* `Perc_LeadsFP_Predicted_Positives`: El porcentaje de *leads* accionables (`FP`) respecto al total de Positivos Predichos en esa audiencia (`TP + FP`). Útil para ver qué parte de la audiencia target serían *leads* accionables.

- `Perc_LeadsFP_Total_Users`: El porcentaje de *leads* accionables (`FP`) respecto al total de usuarios del conjunto train. Útil para ver qué parte del total de usuarios del conjunto train serían *leads* accionables.

Esto dio lugar a los siguientes resultados:

| Audiencia | Umbral_Score | Precision | Recall | Lift (Precision / Prop_P) | Leads_Retargeting_FP | Total_Predicted_Positives | Perc_LeadsFP_Predicted_Positives | Perc_LeadsFP_Total_Users |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VIP (Top 5 %)** | 0.4215 | 0.5061 | 0.1638 | **3.27x** | 244 | 494 | 49.39 % | 2.47 % |
| **Estándar (Top 15 %)** | 0.2867 | 0.4000 | 0.3879 | **2.59x** | 888 | 1480 | 60.00 % | 9.00 % |
| **Amplia (Top 30 %)** | 0.2054 | 0.3258 | 0.6317 | **2.11x** | 1995 | 2959 | 67.42 % | 20.23 % |

Vemos un comportamiento coherente con la lógica de negocio: al aumentar el tamaño de la audiencia disminuyen tanto la proporción de usuarios que sí compraron (`Precision`) como la concentración de intención de compra frente al azar (`Lift`), mientras que aumenta la cobertura total de compradores capturados (`Recall`). **Esto confirma que el modelo ordena eficazmente a los usuarios según su propensión de compra, permitiendo elegir entre campañas más rentables y selectivas (VIP) o estrategias de mayor alcance y bajo coste (Amplia).**

#### 3.4.3. Evaluación de Negocio (OOF Train vs Test Set)

Finalmente, su utilizaron probabilidades predichas en Test con el modelo final (propensity scores) y se aplicarón los umbrales de negocio a las distintas audiencias objetivo del Test Set obteniendo las mismas métricas previas y diferentes gráficos (Curva Precision-Recall, Matrices de Confusión y Gráficos de Barras). Los resultados detallados y gráficos pueden consultarse en el apartado 3.4.2 del notebook correspondiente.

Se obtuvieron las siguientes métricas:

| Audiencia | Umbral_Score | Precision | Recall | Lift (Precision / Prop_P) | Leads_Retargeting_FP | Total_Predicted_Positives | Perc_LeadsFP_Predicted_Positives | Perc_LeadsFP_Total_Users |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VIP (Top 5 %)** | 0.4215 | 0.4130 | 0.1492 | **2.67x** | 81 | 138 | 58.70 % | 3.28 % |
| **Estándar (Top 15 %)** | 0.2867 | 0.4016 | 0.3953 | **2.59x** | 225 | 376 | 59.84 % | 9.12 % |
| **Amplia (Top 30 %)** | 0.2054 | 0.3058 | 0.6204 | **1.97x** | 538 | 775 | 69.42 % | 21.82 % |

En conjunto, al aplicar los umbrales para las dintintas audiencias de negocio sobre el `Test Set` **las métricas mantienen su eficacia y estabilidad** al enfrentarse a datos nuevos que no ha visto nunca, al compararlas con las métricas correspondientes obtenidas con el Train Set.

Por otro lado, la Curva Precision-Recall proporciono 3 zonas bien diferenciadas:

* `Zona de inestabilidad`: En esta parte izquierda de la curva el Recall es muy bajo, mientras que la Precision empieza baja y luego sube mientras va acumulando aciertos hasta llegar a unos **picos en torno a una Precision = 0.6**. Al ser un segmento reducido (por encima del Top 5%), esta zona presenta alta volatilidad y un margen de error amplio. Así que no conviene poner un corte de audiencia demasiado estrecho, pues nos interesa centrarnos en capturar FP que puedan servir como leads para retargeting, y umbrales en esta zona nos dejarían un volumen bajo de usuarios.

- `Zona de meseta central`: En esta parte central encontramos una **zona estable en torno a Precision = 0.4**. Es la zona óptima de negocio donde el modelo encuentra un grupo amplio y homogéneo de usuarios con alto parecido a compradores reales, donde sube el Recall manteniendo la Precision estable, lo que permite captar distintos volumenes de negocio sin sacrificar calidad. Esta meseta presenta un Lift 2.58X, con una propensión de compra 2.5 veces superior al azar, siendo el 60% restante de usuarios (FP) un grupo de no-compradores con un comportamiento de comprador. **Nuestros cortes del Top 5% y 15% caen en esta zona, demostrando que son puntos de corte operativos consistentes y de utilidad**.

* `Zona de caida`: En esta parte derecha, a **partir de Recall = 0.4, se observa una zona decreciente** donde aumentar Recall implica una caida progresiva de la Precision. Al ampliar la audiencia sube el volumen, pero el modelo empieza a incluir usuarios dudosos, reduciendo la calidad de la audiencia. **Nuestro corte del Top 30% cae en esta zona, por lo que puede ser interesante para retargeting de coste marginal cero**.

## 4. Conclusiones Finales

- El modelo final (`LightGBM Tuned`) generaliza de forma estable, más que duplicando la tasa base de azar y manteniendo una calibración fiable sin signos de *overfitting* ni *underfitting* severo.

En conjunto, vemos que el rendimiento del modelo optimizado (`Tuned`) se mantiene estable con pequeñas variaciones al comparar las validaciones cruzadas previas (`Train CV`) y el conjunto final de test (`Test`), lo que indicaría ausencia de sobreajuste significativo, una buena generalización y alta estabilidad del modelo final.


* El análisis de *Feature Importance* y *SHAP* revela que las variables relacionadas con exploración de productos (`ProductRelated`), las tasas de salida (`ExitRates`) y la estacionalidad mensual (`Month`) son los factores que más impulsan o penalizan al *propensity score*. Lo que proporciona información que en algunos casos podría ser accionable.

- El modelo junto con sus *propensity scores* permitiría poner en marcha campañas de retargeting según distintos presupuestos con los leads potenciales obtenidos: la **Audiencia VIP (Top 5 %)** maximiza eficiencia, la **Estándar (Top 15 %)** ofrece el mejor equilibrio eficiencia-volumen  y la **Amplia (Top 30 %)** priorizando el alcance.

# 5. Limitaciones

La naturaleza del propio dataset genera ciertas limitaciones del modelo que podrían afectar a sus predicciones y su operativa, por lo que debe usarse con cautela en determinados escenarios:

* La variable `ExitRates` tiene valores máximos de 0.2 lo que podría indicar un tope artificial. Por lo que se ha podido investigar es efectivamente un tope (cap) aplicado en el preprocesamiento o anonimización del dataset, y no una limitación nativa de Google Analytics. De manera que el modelo no se ha enfrentado a valores mayores de este tope.

- La variable `Month` solo tiene representados 10 de los 12 meses posibles. Por ello, el modelo no se ha enfrentado a sesiones de los meses de enero y abril.

* La variable `TrafficType` al estar codificada carece de significado interpretable, ya que no sabemos lo que significa cada categoría, así que no podemos proponer acciones concretas dirigidas a los distintos canales de tráfico por una falta de contexto.

- La ausencia de NaNs en el dataset. De manera que el modelo no se ha enfrentado ni a valores NaN ni a datos imputados. Esto también afecta al propio `ColumnTransformer` desarrollado, por simplicidad se añadió un `SimpleImputer con strategy='median'` de cara a futura puesta en producción para que cuando entren NaNs no diera error, pero habría que programar alguna forma de marcar esas predicciones subóptimas.

Además, debido al desbalanceo del dataset original (solo 15.5% de las sesiones acababan en compra) y la decisión de no aplicar re-balanceo artificial para preservar la calibración, el modelo puede ser conservador en algunos casos, en especial en umbrales altos de negocio, como la Audiencia Top 5%.

Por otro lado, ha quedado pendiente el despliegue del modelo en Streamlit (u otra plataforma similar) para facilitar su uso operativo en fase de producción.

# 6. Perspectivas Futuras

La forma de evaluar la calidad real de esos leads FP sería realizar lo que se conoce en marketing como un Test A/B con un grupo control en el que no se impacta a los leads (se les excluye de la campaña de retargeting) y compararlo con los leads FP que sí fueron impactados. Además, al trabajar con 3 audiencias con distintas estrategias de retargeting, habría que definir 3 tramos claros para cada estrategia: Audiencia VIP (Top 5%), Estándar (Top 5% -15%) y Amplia (Top 15%-30%), con audiencias mutuamente excluyentes. Es decir, simplemente aplicar la regla operativa de excluir de cada nivel superior al siguiente y reservar un grupo de control para cada tramo. Además, habría que medir el retorno incremental real (ROI) de estas estrategias.