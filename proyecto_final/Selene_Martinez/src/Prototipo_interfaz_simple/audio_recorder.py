import sounddevice as sd
import wave
import numpy as np  # Aunque no se usa directamente aquí, podría ser útil en el futuro

def grabar_audio(duracion=2, frecuencia_muestreo=44100):
    """Graba audio del micrófono."""
    print("Grabando...")
    grabacion = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='int16')
    sd.wait()
    return grabacion, frecuencia_muestreo

def guardar_audio(grabacion, frecuencia_muestreo, nombre_archivo="grabacion.wav"):
    """Guarda la grabación en un archivo WAV."""
    with wave.open(nombre_archivo, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes para int16
        wf.setframerate(frecuencia_muestreo)
        wf.writeframes(grabacion.tobytes())
    print(f"Audio guardado como {nombre_archivo}")

if __name__ == '__main__':
    grabacion, frecuencia = grabar_audio()
    guardar_audio(grabacion, frecuencia)