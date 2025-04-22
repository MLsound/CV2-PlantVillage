# CV2-PlantVillage
# Trabajo práctico integrador - Visión por Computadoras II
## Carrera de Especialización en Inteligencia Artificial - Cohorte 17

### Autores:
* Piñero, Juan Cruz
* Lloveras, Alejandro
* Méndez, Diego Martín

### Objetivo del trabajo
El objetivo de este trabajo práctico integrador es procesar **54305** imágenes de hojas, pertenecientes a **14 especies** de plantas, utilizando modelos de *Computer Vision* para clasificar entre plantas saludables y múltiples enfermedades (**38 clases en total**).

---

### Estructura del Repositorio

La estructura del repositorio se organiza de la siguiente manera:

```
.
├── Documentacion/                      # Documentos relacionados con los avances del proyecto
├── Experimentos/                       # Resultados y artefactos de los diferentes experimentos
│   ├── experimento_cnn/                # Detalles del Experimento 01 _(CNN Model)_
│   ├── experimento_vgg/                # Detalles del Experimento 02 _(VGG Model)_
│   ├── experimento_a1/                 # Detalles del Experimento A1 _(estrategia 1)_
│   ├── experimento_a2/                 # Detalles del Experimento A2 _(estrategia 2)_ah
│   ├── experimento_gray/               # Detalles del Experimento para dataset en escala de grises.
│   └── experimento_segmented/          # Detalles del Experimento para dataset segmentado.
├── Versiones Colab/                    # Notebooks utilizados y/o probados en Google Colab
│   ├── DataAugmentation.ipynb
│   ├── DataAugmentation2.ipynb
│   └── TrainExperiments.ipynb
├── .gitignore                          # Archivo para ignorar archivos y carpetas en Git
├── 1 Exploracion y Split.ipynb         # Notebook para la exploración inicial de los datos (EDA) y división en conjuntos de train/validation/test.
├── 2 Data Augmentation.ipynb           # Notebook con la implementación de estrategias de aumentación de datos.
├── 3 Modelo CNN.ipynb                  # Notebook con la implementación y entrenamiento de un modelo CNN base.
├── 4 Modelo VGG.ipynb                  # Notebook con la implementación y entrenamiento de modelos VGG.
├── 5 Model Evaluation.ipynb            # Notebook para la evaluación final de los modelos entrenados.
├── architectures.py                    # Script con definiciones de arquitecturas de modelos
├── constants.yaml                      # Archivo de configuración con constantes utilizadas en scripts y notebooks.
├── CNN_4_blocks_history.pkl            # Historial de entrenamiento del modelo CNN con 4 bloques.
├── data_utils.py                       # Script con funciones utilitarias para la carga, preprocesamiento y manejo de datos.
├── dataframe_augmented.csv             # Archivo CSV con información sobre datos aumentados _(estrategia 1)_
├── dataframe_augmented2.csv            # Archivo CSV con información sobre datos aumentados _(estrategia 2)_
├── dataframe_segmented_splitted.csv    # Archivo CSV con información sobre los datos segmentados divididos (train/val/test)
├── dataframe_segmented.csv             # Archivo CSV con información inicial del dataset segmentado
├── dataframe_splitted.csv              # Archivo CSV con información sobre los datos divididos (train/val/test)
├── dataframe.csv                       # Archivo CSV con información inicial del dataset
├── dataloading.py                      # Script con funciones para la carga eficiente de datos, potencialmente usando DataLoader
├── eda_utils.py                        # Script con funciones utilitarias para el análisis exploratorio de datos (EDA)
├── environment.yml                     # Archivo para definir el entorno de Conda
├── model_evaluation.py                 # Script con funciones para realizar la evaluación de modelos (métricas, reportes).s.
├── model_vgg_2_history.pkl             # Historial de entrenamiento del modelo VGG (variante 2).
├── model_vgg_4_history.pkl             # Historial de entrenamiento del modelo VGG (variante 4).
├── model_vgg_6_history.pkl             # Historial de entrenamiento del modelo VGG (variante 6).
├── model_vgg_10_history.pkl            # Historial de entrenamiento del modelo VGG (variante 10).
├── README.md                           # Este archivo
└── representative_histogram.csv        # Archivo CSV relacionado con análisis de histogramas de imágenes.
```

### Dataset

El dataset utilizado en este proyecto consiste en **54305** imágenes de hojas de **14 especies** de plantas, presentando **38 clases** que corresponden a diferentes estados de salud (saludable y varias enfermedades). Se descarga automáticamente en local desde Kaggle y es almacenado en una carpeta oculta asignada por el sistema. La información detallada del dataset se encuentra registrada en los archivos CSV (`dataframe.csv`, `dataframe_splitted.csv`, etc.) y las imágenes están ubicadas en el directorio especificado en el archivo de configuración `constants.yaml`.

### Configuración de Rutas (constants.yaml)

El archivo `constants.yaml` contiene las rutas absolutas utilizadas por el proyecto en local para acceder al dataset, los datos aumentados y los datos divididos. Especifica el valor de las constantes: `AUG_PATH`, `DATASETS_ROOT`, `DATASET_PATH`, `ROOT_DIR` y `SPLITTED_PATH`; las cuales permiten encontrar los archivos en el entorno de ejecución correspondiente, lo que permite utilizar los mismos scripts en Colab y en nuestros equipos.

### Exploración de Datos (EDA)

El análisis exploratorio de datos se lleva a cabo en el notebook `1 Exploracion.ipynb`. Este notebook utiliza funciones definidas en `eda_utils.py` para comprender la distribución de las clases, características de las imágenes, y cualquier otro aspecto relevante del dataset antes del modelado.

### Aumentación de Datos

En el notebook `2 Data Augmentation.ipynb` se exploran técnicas de aumentación de datos para enriquecer el dataset y mejorar la robustez de los modelos. 
`DataAugmentation.ipynb` y `DataAugmentation2.ipynb` en `Versiones Colab/`, junto con los archivos `dataframe_augmented.csv` y `dataframe_augmented2.csv`, documentan este proceso, aplicando dos estrategias de aumentación diferentes.

### Modelos y Arquitecturas

Las arquitecturas de los modelos de *Computer Vision* implementados se definen en el script `architectures.py`. El notebook `3 Modelo CNN.ipynb` presenta la implementación y entrenamiento de un modelo CNN base.

### Entrenamiento de Modelos

El proceso de entrenamiento de los modelos se documenta en los notebooks de experimentos dentro del directorio `Experimentos/` y en `TrainExperiments.ipynb` dentro de `Versiones Colab/`. Los historiales de entrenamiento se guardan en formatos JSON y Pickle.

### Versiones Colab
Esta carpeta contiene versiones simplificadas de los notebooks pensadas exclusivamente para ejecutar partes específicas del proceso en el entorno de Colab, utilizando como almacenamiento una carpeta compartida de Google Drive.

### Experimentos

El directorio `Experimentos/` almacena los resultados y artefactos de las diferentes corridas experimentales. Cada subdirectorio (e.g., `experimento_01/`) contiene el modelo entrenado (`best_model.keras`), el notebook específico del experimento, y los historiales de entrenamiento.

### Evaluación de Modelos

La evaluación del rendimiento de los modelos se realiza utilizando el notebook `model_evaluation.ipynb` y las funciones proporcionadas en el script `model_evaluation.py`.

---