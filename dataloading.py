import pandas as pd
import tensorflow as tf


import os, re

import yaml


def load_env_variables():
    # Cargar variables desde el archivo YAML
    try:
        # Verificar si el archivo YAML existe
        yaml_filename = "constants.yaml"
        with open(yaml_filename, "r") as yaml_file:
            constants_data = yaml.safe_load(yaml_file)

        # Acceder a las variables
        ROOT_DIR = constants_data.get("ROOT_DIR")
        DATASET_PATH = constants_data.get("DATASET_PATH")
        SPLITTED_PATH = constants_data.get("SPLITTED_PATH")

        print(f"✅ Se han cargado las variables de configuración desde '{yaml_filename}'")
        print(f" - ROOT_DIR: {ROOT_DIR}")
        print(f" - DATASET_PATH: {DATASET_PATH}")
        print(f" - SPLITTED_PATH: {SPLITTED_PATH}")
    except FileNotFoundError:
        print(f"Error: El archivo 'constants.yaml' no se encontró en la ubicación actual: {os.getcwd()}")
        print("Se creará nuevamente al correr el notebook.")
        ROOT_DIR = None
        DATASET_PATH = None
        SPLITTED_PATH = None
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo YAML: {e}")
    return constants_data
# Busca la carpeta raíz del dataset en el directorio donde fue descargado
def find_path(folder):
    DATASET_PATH = load_env_variables().get("DATASET_PATH")
    match = re.search(fr"^(.*?)/{folder}/", DATASET_PATH)
    if match:
        prefix = match.group(1)
        path = os.path.join(prefix, f"{folder}/")
        return path
    else:
        print(f'No se ha podido encontrar la carpeta "{folder}" en {DATASET_PATH}')
        return None
# Guarda directorio del dataset dividido
path = find_path("plantvillage-dataset")
SPLITTED_PATH = f"{path}splitted/" if path else None
def load_dataloaders(preprocess_function=None):# Carga el dataset de imágenes desde el directorio especificado
    train_images = ""; test_images = ""; valid_images = ""
    df_split = pd.read_csv('dataframe_splitted.csv').set_index('id')
    splits = df_split['split'].value_counts().index.tolist()
    print("Cargando datasets desde el directorio…\n")
    for split in splits:
        data_folder = f'{SPLITTED_PATH}{split}/'

        # Carga el conjunto de datos desde el directorio especificado
        # Utiliza la función de TensorFlow para crear un dataset de imágenes
        match split:
            case 'train':
                print(f"Cargando dataset de entrenamiento desde:\n > {data_folder}")
                train_images,class_names_train = load_from_directory(data_folder,pretrained_net_preprocess = preprocess_function)
            case 'test':
                print(f"Cargando dataset de test desde:\n > {data_folder}")
                test_images,_ = load_from_directory(data_folder,pretrained_net_preprocess=preprocess_function)
            case 'valid':
                print(f"Cargando dataset de validación desde:\n > {data_folder}")
                valid_images , _= load_from_directory(data_folder,pretrained_net_preprocess=preprocess_function)
            case _: # En caso de no coincidir con ninguno de los splits
                print(f"⚠️ El split '{split}' no es reconocido. No se cargará ningún dataset.")
                continue # Salta al siguiente split
        print(f"✅ Dataset cargado exitosamente.\n")
    return {'train':train_images,'test':test_images,'valid':valid_images},class_names_train

# Data laoders setup
def load_from_directory(data_folder,pretrained_net_preprocess = None):
    """
    Carga un dataset de imágenes desde un directorio específico.

    Args:
        data_folder (str): Ruta al directorio que contiene las imágenes.

    Returns:
        tf.data.Dataset: Dataset de TensorFlow con las imágenes y etiquetas.
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_folder,  # Ruta al directorio de datos
        labels="inferred",  # Las etiquetas se infieren automáticamente desde los nombres de las carpetas
        label_mode="categorical",  # Las etiquetas se codifican como categorías (one-hot encoding)
        class_names=None,  # Las clases se detectan automáticamente
        color_mode="rgb",  # Las imágenes se cargan en modo RGB
        batch_size=128,  # Tamaño de lote para el entrenamiento
        image_size=(256, 256),  # Redimensiona las imágenes a 128x128 píxeles
        shuffle=True,  # Mezcla las imágenes aleatoriamente
        seed=42,  # No se utiliza una semilla específica para la aleatorización
        validation_split=None,  # No se realiza una división de validación aquí
        subset=None,  # No se especifica un subconjunto (train/validation)
        interpolation="bilinear",  # Método de interpolación para redimensionar las imágenes
        follow_links=False,  # No sigue enlaces simbólicos
        crop_to_aspect_ratio=False  # No recorta las imágenes para ajustar la relación de aspecto
    )
    class_names = dataset.class_names
    if(pretrained_net_preprocess):
        dataset = dataset.map(pretrained_net_preprocess)

    return dataset, class_names