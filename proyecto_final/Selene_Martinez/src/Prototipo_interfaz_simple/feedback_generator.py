from transformers import pipeline
import os
import numpy as np  # Necesitamos numpy aquí

modelo_path = "src/gpt2-pron-feedback"  # Asegúrate de que la ruta sea correcta

def generar_feedback(input_word, phoneme, f1_usuario, f2_usuario, datos_referencia):
    """Genera retroalimentación sobre la pronunciación usando un modelo GPT-2."""
    try:
        generator = pipeline("text-generation", model=modelo_path, tokenizer=modelo_path)
        prompt = (
            f"El usuario intentó pronunciar la vocal /{phoneme}/ como en la palabra \"{input_word}\".\n"
            f"Según mi análisis, la primera frecuencia de su voz (similar a qué tan abierta es la boca) fue de {int(f1_usuario)} Hz, "
            f"y la segunda frecuencia (similar a dónde está posicionada la lengua en la boca) fue de {int(f2_usuario)} Hz.\n"
            f"Para la vocal /{phoneme}/, las frecuencias usuales suelen estar alrededor de {int(np.mean(datos_referencia[:, 0]))} Hz para la primera "
            f"y {int(np.mean(datos_referencia[:, 1]))} Hz para la segunda.\n"
            f"Considerando esta información, brinda un consejo breve y amigable para ayudar al usuario a mejorar su pronunciación de esta vocal. "
            f"El consejo debe ser fácil de entender y enfocado en cómo podría sentir o mover su boca y lengua. Evita usar términos técnicos como 'F1' o 'F2'."
        )
        respuesta = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.8)[0]["generated_text"]
        return respuesta
    except Exception as e:
        return f"Error al generar retroalimentación: {e}"

if __name__ == '__main__':
    # Ejemplo de prueba (necesitarías datos de referencia simulados para probar completamente)
    datos_ref_ae = np.array([[700, 1200], [650, 1150]])
    feedback = generar_feedback("cat", "æ", 750, 1250, datos_ref_ae)
    print(feedback)