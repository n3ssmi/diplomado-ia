import numpy as np
from scipy.spatial.distance import euclidean

def calcular_distancia_formantes(formantes_usuario, formantes_referencia):
    """Calcula la distancia euclidiana promedio entre los formantes del usuario y los de referencia."""
    if formantes_usuario is None or formantes_referencia is None or formantes_referencia.size == 0:
        return float('inf')  # Indica una gran diferencia si no hay datos

    f1_usuario, f2_usuario = formantes_usuario
    distancias = []
    for f1_ref, f2_ref in formantes_referencia:
        distancia = np.sqrt((f1_usuario - f1_ref)**2 + (f2_usuario - f2_ref)**2)
        distancias.append(distancia)
    return np.mean(distancias)

def comparar_formantes(formantes_usuario, formantes_referencia):
    """Calcula la distancia promedio entre los formantes del usuario y la referencia."""
    distancias = [euclidean(formantes_usuario, ref) for ref in formantes_referencia]
    return np.mean(distancias)

def obtener_score_formantes(distancia_promedio):
    """Calcula un score basado en la distancia promedio.
    Este es un ejemplo y puede necesitar ajuste según tus datos."""
    max_distancia_aceptable = 500  # Ajusta este valor
    if distancia_promedio > max_distancia_aceptable:
        return 0
    else:
        # Un score lineal inverso a la distancia (ejemplo)
        score = 100 - (distancia_promedio / max_distancia_aceptable) * 100
        return max(0, min(100, score)) # Asegurar que el score esté entre 0 y 100