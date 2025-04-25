import numpy as np
import os

# Carpeta donde se guardarán los archivos .npy
output_dir = "formantes_referencia"
os.makedirs(output_dir, exist_ok=True)

# Valores de referencia aproximados (F1, F2) en Hz para vocales
# Basado en estudios fonéticos de hablantes nativos de inglés
referencias_vocales = {
    "ae": {  # /æ/ como en "cat"
        "nombre": "ae",
        "valores": np.array([
            [700, 1500],
            [650, 1450],
            [720, 1530],
            [710, 1490]
        ])
    },
    "ih": {  # /ɪ/ como en "bit"
        "nombre": "ih",
        "valores": np.array([
            [400, 1900],
            [390, 1850],
            [410, 2000],
            [420, 1950]
        ])
    },
    "uh": {  # /ʌ/ como en "cut"
        "nombre": "uh",
        "valores": np.array([
            [600, 1200],
            [580, 1150],
            [610, 1250],
            [590, 1180]
        ])
    }
}

# Guardar los datos y estadísticas
for clave, datos in referencias_vocales.items():
    matriz = datos["valores"]
    nombre_archivo = datos["nombre"]
    
    # Guardar matriz original
    np.save(os.path.join(output_dir, f"referencia_{nombre_archivo}.npy"), matriz)
    
    # Calcular y guardar media y desviación estándar
    media = np.mean(matriz, axis=0)
    std = np.std(matriz, axis=0)
    
    np.save(os.path.join(output_dir, f"media_{nombre_archivo}.npy"), media)
    np.save(os.path.join(output_dir, f"std_{nombre_archivo}.npy"), std)

    print(f"Guardados archivos para: {nombre_archivo}")
