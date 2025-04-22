# data_utils.py
# FUNCIONES DE UTILIDAD PARA GESTIÓN DE DATOS Y MANEJO DE ARCHIVOS
# use: from data_utils import <function_name>

#Librerías necesarias
import shutil
import os
import yaml
import pandas as pd
from PIL import Image


def test():
    print("Importación exitosa")

def find_folder(path):
    return os.path.basename(os.path.dirname(path))
    
def dataset_already_exists(path_to_check: str) -> bool | None:
    """
    Verifica si el directorio especificado existe y está vacío.

    Args:
        path_to_check (str): Ruta del directorio a verificar.

    Returns:
        bool: True si el directorio existe y está vacío, False en caso contrario.
    """
    if not os.path.exists(path_to_check):
        # El directorio no existe -> Crea el directorio
        #print(f"☑️ El directorio no existe, aún no ha sido creado:\n > {path_to_check}") # Debugging
        return False # No realiza ninguna acción
    else:
        # Verificar si el directorio está vacío
        try:
            # Explora el contenido del directorio
            content = os.listdir(path_to_check)
            #print(content) # Debugging

            # Si el directorio está vacío, se puede eliminar directamente
            #       -> Elimina sin confirmación
            if not content:
                os.rmdir(path_to_check) # Elimina el directorio vacío
                print(f"☑️ El directorio estaba vacío y se ha eliminado de forma automática:\n > {path_to_check}\n")
                return False
            
            # Si el directorio contiene sólo archivos ocultos (de sistema)
            #       -> Elimina sin confirmación
            elif all([file.startswith('.') for file in content]):
                shutil.rmtree(path_to_check) # Elimina el directorio y su contenido
                print(f"☑️ El directorio sólo contenía archivos ocutlos, por lo que se ha eliminado de forma automática:\n > {path_to_check}\n")
                return False

            # Si hay archivos visibles en el directorio (dataset ya existe)
            #       -> Solicita permiso para eliminarlos
            else:
                # Input de confirmación del usuario
                confirmacion = input(f"⚠️ El directorio especificado ya existe y contiene archivos. ¿Deseas eliminar todo su contenido y el directorio en sí? [Y/N]: '{path_to_check}'").strip().lower()
                # Verifica la respuesta del usuario
                if confirmacion == 'y':
                    shutil.rmtree(path_to_check) # Elimina el directorio y su contenido
                    print(f"✅ El directorio y su contenido han sido eliminados exitosamente:\n > {path_to_check}\n")
                    return False
                else:
                    print(f"⛔️ La eliminación del directorio ha sido denegada por el usuario:\n  > {path_to_check}")
                    return True
        
        except OSError as e:
            print(f"❌ Error al eliminar el directorio vacío en {path_to_check}: {e}\n")
            return None
        except Exception as e:
            print(f"‼️ Ocurrió un error inesperado al intentar eliminar el directorio vacío en {path_to_check}: {e}\n")
            return None


def import_dataset(filename: str = 'dataframe_splitted.csv') -> pd.DataFrame | None:
    """
    Carga un DataFrame desde un archivo CSV y define la columna 'id' como índice.

    Args:
        filename (str): Nombre del archivo CSV que contiene los datos. 
                        Por defecto, 'dataframe_splitted.csv'.

    Returns:
        pandas.DataFrame: El DataFrame cargado con 'id' como índice, o None si ocurre un error.
    """
    try:
        df = pd.read_csv(filename).set_index('id')
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no se encontró en la ubicación actual: {os.getcwd()}")
        print("Se creará nuevamente al correr las celdas de 'Importación de imágenes'.")
        return None
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo CSV: {e}")
        return None

# Para cargar rutas de acceso a datos
# ROOT_DIR, DATASETS_ROOT, DATASET_PATH, SPLITTED_PATH, AUG_PATH = data_utils.import_from_yaml()
def import_from_yaml():
    """
    Importa las constantes de configuración desde un archivo YAML llamado 'constants.yaml'.

    Returns:
        tuple: Una tupla con las constantes ROOT_DIR, DATASETS_ROOT, DATASET_PATH, SPLITTED_PATH, AUG_PATH.
    """
    try:
        # Verificar si el archivo YAML existe
        yaml_filename = "constants.yaml"
        with open(yaml_filename, "r") as yaml_file:
            constants_data = yaml.safe_load(yaml_file)

        # Acceder a las variables
        ROOT_DIR = constants_data.get("ROOT_DIR")
        DATASETS_ROOT = constants_data.get("DATASETS_ROOT")
        DATASET_PATH = constants_data.get("DATASET_PATH")
        SPLITTED_PATH = constants_data.get("SPLITTED_PATH")
        AUG_PATH = constants_data.get("AUG_PATH")

        print(f"✅ Se han cargado las variables de configuración desde '{yaml_filename}'")
        print(f" - ROOT_DIR: {ROOT_DIR}")
        print(f" - DATASETS_ROOT: {DATASETS_ROOT}")
        print(f" - DATASET_PATH: {DATASET_PATH}")
        print(f" - SPLITTED_PATH: {SPLITTED_PATH}")
        print(f" - AUG_PATH: {AUG_PATH}")
        return ROOT_DIR, DATASETS_ROOT, DATASET_PATH, SPLITTED_PATH, AUG_PATH

    except FileNotFoundError:
        print(f"Error: El archivo 'constants.yaml' no se encontró en la ubicación actual: {os.getcwd()}")
        print("Se creará nuevamente al correr el notebook.")
        ROOT_DIR = None
        DATASETS_ROOT = None
        DATASET_PATH = None
        SPLITTED_PATH = None
        AUG_PATH = None
        return ROOT_DIR, DATASETS_ROOT, DATASET_PATH, SPLITTED_PATH, AUG_PATH
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo YAML: {e}")


# Para guardar rutas de acceso a datos
# data_utils.save_into_yaml(ROOT_DIR, DATASETS_ROOT, DATASET_PATH, SPLITTED_PATH, AUG_PATH)
def save_into_yaml(ROOT_DIR: str | None = None,
                   DATASETS_ROOT: str | None = None,
                   DATASET_PATH: str | None = None,
                   SPLITTED_PATH: str | None = None,
                   AUG_PATH: str | None = None):
    """
    Actualiza las variables de configuración en un archivo YAML ('constants.yaml'),
    solo guardando aquellas que no son None y manteniendo las existentes.

    Args:
        ROOT_DIR (str): Directorio raíz (se guarda si no es None).
        DATASETS_ROOT (str): Directorio de datasets (se guarda si no es None).
        DATASET_PATH (str): Ruta del dataset (se guarda si no es None).
        SPLITTED_PATH (str): Ruta de datos divididos (se guarda si no es None).
        AUG_PATH (str): Ruta de datos aumentados (se guarda si no es None).
    """
    yaml_filename = "constants.yaml"
    existing_data = {}

    # 1. Read existing data from the YAML file if it exists
    try:
        if os.path.exists(yaml_filename):
            with open(yaml_filename, "r") as yaml_file:
                loaded_data = yaml.safe_load(yaml_file)
                # yaml.safe_load returns None for an empty file
                if loaded_data is not None:
                    existing_data = loaded_data
                print(f"ℹ️ Existing data loaded from '{yaml_filename}'")
        else:
             print(f"ℹ️ '{yaml_filename}' not found. Starting with empty data.")
    except yaml.YAMLError as e:
        print(f"❌ Error reading YAML file '{yaml_filename}': {e}. Starting with empty data.")
        existing_data = {} # Ensure existing_data is a dict even if reading fails
    except Exception as e:
         print(f"❌ An unexpected error occurred while reading '{yaml_filename}': {e}. Starting with empty data.")
         existing_data = {}


    # 2. Determine which variables from arguments are not None
    updates_to_apply = {
        key: value for key, value in {
            "ROOT_DIR": ROOT_DIR,
            "DATASETS_ROOT": DATASETS_ROOT,
            "DATASET_PATH": DATASET_PATH,
            "SPLITTED_PATH": SPLITTED_PATH,
            "AUG_PATH": AUG_PATH
        }.items() if value is not None and isinstance(value, str)
    }

    # 3. If no non-None values were provided, just exit
    if not updates_to_apply:
        print("⚠️ No variables with non-None values provided for update. File not changed.")
        return

    # 4. Merge the updates into the existing data
    # This will add new keys or overwrite existing ones with the non-None values
    existing_data.update(updates_to_apply)
    print(f"🔄 Applying updates: {updates_to_apply}")

    # 5. Write the merged, updated data back to the file
    try:
        with open(yaml_filename, "w") as yaml_file:
            yaml.dump(existing_data, yaml_file, default_flow_style=False, sort_keys=False) # Added sort_keys=False for potentially better diffs
        print(f"✅ Se han actualizado/guardado las variables de configuración en '{yaml_filename}'")
    except Exception as e:
        print(f"❌ Error writing updated YAML file '{yaml_filename}': {e}")

# Carga de imagenes en memoria y visualización
def load_image(data: pd.DataFrame, index: int, root: str):
    """
    Carga una imagen PIL desde una fila específica de un DataFrame.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que contiene las rutas de las imágenes.
        index (int): El índice de la fila en el DataFrame para cargar la imagen.
        root_dir (str): El directorio raíz donde se encuentran las imágenes.

    Returns:
        PIL.Image.Image: La imagen cargada como un objeto PIL.Image, o None si ocurre un error.
    """

    # if index < 0 or index >= len(data):
    #     print("Índice fuera de rango.")
    #     return None

    row = data.iloc[index]
    relative_path = row['image_path']
    filename = row['filename']
    full_path = os.path.join(root, relative_path, filename)

    try:
        img = Image.open(full_path)
        return img
    except FileNotFoundError:
        print(f"Archivo no encontrado: {full_path}")
        return None
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None
    
# Carga de imagenes en memoria y visualización
def load_image_idx(data, root: str):
    """
    Carga una imagen PIL desde una fila específica de un DataFrame.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que contiene las rutas de las imágenes.
        index (int): El índice de la fila en el DataFrame para cargar la imagen.
        root_dir (str): El directorio raíz donde se encuentran las imágenes.

    Returns:
        PIL.Image.Image: La imagen cargada como un objeto PIL.Image, o None si ocurre un error.
    """
    # if index < 0 or index >= len(data):
    #     print("Índice fuera de rango.")
    #     return None

    row = data
    relative_path = row['image_path']
    filename = row['filename']
    full_path = os.path.join(root, relative_path, filename)

    try:
        img = Image.open(full_path)
        return img
    except FileNotFoundError:
        print(f"Archivo no encontrado: {full_path}")
        return None
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None