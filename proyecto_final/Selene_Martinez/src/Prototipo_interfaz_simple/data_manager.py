import numpy as np
import os
import soundfile as sf
import csv

# Obtiene el directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "formantes_referencia")
CMU_ARTIC_WAV_DIR = os.path.join(DATA_DIR, "CMU_Artic") # La carpeta principal de los WAVs

# Diccionario que mapea el nombre del acento al archivo CSV
ACCENT_MAPPING_CSV = {
    "US_BDL": os.path.join(DATA_DIR, "phonemas_cmu_us_bdl_arctic.csv"), # Ejemplo de nombre
    "US_JMK": os.path.join(DATA_DIR, "phonemas_cmu_us_awb_arctic.csv"), # Otro ejemplo
    # Añade más acentos y sus archivos CSV correspondientes
}

def cargar_referencia(nombre_archivo_relativo):
    nombre_archivo = os.path.join(DATA_DIR, nombre_archivo_relativo)
    try:
        return np.load(nombre_archivo, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Archivo de referencia '{nombre_archivo}' no encontrado.")
        return None

def obtener_audio_correcto_cmu(sonido, acento="US_BDL"): # Añade el argumento 'acento' con un valor por defecto
    """Intenta obtener un archivo de audio del CMU Artic que contenga el sonido para el acento dado."""
    audio_file = None
    csv_path = ACCENT_MAPPING_CSV.get(acento)
    if not csv_path:
        print(f"Error: No se encontró el archivo CSV para el acento '{acento}'.")
        return None

    wav_accent_dir = os.path.join(CMU_ARTIC_WAV_DIR, f"cmu_{acento.lower()}_arctic", "wav") # Ajusta la estructura si es diferente

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Saltar la primera línea si tiene encabezado
            for row in reader:
                if len(row) >= 3 and row[1] == sonido[1:-1]: # Compara el fonema sin las barras
                    audio_file = row[2] + ".wav"
                    break # Tomamos el primer archivo que encontramos
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV de mapeo: {csv_path}")
        return None
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return None

    if audio_file:
        audio_path = os.path.join(wav_accent_dir, audio_file)
        if os.path.exists(audio_path):
            return audio_path
        else:
            print(f"Error: No se encontró el archivo de audio del CMU Artic: {audio_path}")
            return None
    else:
        print(f"Advertencia: No se encontró el fonema '{sonido}' en el CSV para el acento '{acento}'.")
        return None

if __name__ == '__main__':
    # Ejemplo de cómo cargar datos de referencia
    referencia_ae = cargar_referencia("referencia_ae.npy")
    if referencia_ae is not None:
        print("Datos de referencia para /æ/ cargados.")

    # Ejemplo de cómo obtener la ruta del audio correcto para un acento específico
    audio_path_ae_bdl = obtener_audio_correcto_cmu("/æ/", "US_BDL")
    if audio_path_ae_bdl:
        print(f"Ruta del audio correcto para /æ/ (US_BDL): {audio_path_ae_bdl}")

    audio_path_ae_jmk = obtener_audio_correcto_cmu("/æ/", "US_JMK")
    if audio_path_ae_jmk:
        print(f"Ruta del audio correcto para /æ/ (US_JMK): {audio_path_ae_jmk}")