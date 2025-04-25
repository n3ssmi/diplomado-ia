import parselmouth
import os
recording_path = os.path.join("data", "records")

def obtener_formantes(archivo_wav):
    """Extrae los formantes F1 y F2 de un archivo WAV."""
    try:
        snd = parselmouth.Sound(archivo_wav)
        formant = snd.to_formant_burg()
        f1 = formant.get_value_at_time(1, 0.05)
        f2 = formant.get_value_at_time(2, 0.05)
        return f1, f2
    except Exception as e:
        print(f"Error al procesar el archivo {archivo_wav}: {e}")
        return None, None

if __name__ == '__main__':
    # Ejemplo de cómo usar la función para un archivo específico
    archivo_prueba = os.path.join(recording_path, "grabacion_æ.wav")
    if os.path.exists(archivo_prueba):
        f1, f2 = obtener_formantes(archivo_prueba)
        if f1 is not None and f2 is not None:
            print("Formantes extraídos (F1, F2):", f1, f2)
        else:
            print("No se pudieron extraer los formantes del archivo de prueba.")
    else:
        print(f"El archivo de prueba {archivo_prueba} no existe. Asegúrate de tener una grabación.")
        