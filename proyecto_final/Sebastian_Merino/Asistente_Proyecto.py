# proyecto_final.py

"""
Proyecto Final - Asistente con GPT-2 + RAG
Autor: Rodrigo  Zeferino y Sebastian Merino
Descripción:
Este script crea un asistente que responde preguntas técnicas usando información
recuperada desde Wikipedia y un archivo personalizado. Luego, genera respuestas
con un modelo GPT-2 que fue fine-tuned con la misma base lógica.
"""

import os
import numpy as np
import faiss
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)

import wikipedia

# ================================
# CONFIGURACIÓN GLOBAL
# ================================
BASE_PATH = Path(".")
BASE_LOGIC_FILE = BASE_PATH / "base_logic_en.txt"
CUSTOM_FILE = BASE_PATH / "base_logica_prueba.txt"
TRAINED_MODEL_PATH = BASE_PATH / "fine_tuned_gpt2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

WIKI_TOPICS = [
    "Market analysis",
    "Consumer behaviour",
    "Marketing",
    "Psychology of marketing",
    "Customer satisfaction",
    "Segmentation (marketing)"
]

# ================================
# FASE 0: Descargar información desde Wikipedia
# ================================
def generar_base_desde_wikipedia():
    print("🌐 Obteniendo información de Wikipedia...")
    contenido = ""
    for topic in WIKI_TOPICS:
        try:
            texto = wikipedia.page(topic).content
            contenido += f"\n\n# {topic}\n" + texto
        except Exception as e:
            print(f"❌ No se pudo obtener '{topic}': {e}")

    with open(BASE_LOGIC_FILE, "w", encoding="utf-8") as f:
        f.write(contenido)

    print(f"✅ Información guardada en: {BASE_LOGIC_FILE}\nCaracteres totales: {len(contenido)}")

# ================================
# FASE 1: Fine-tuning de GPT-2
# ================================
def finetune_gpt2():
    print("\n🔧 Entrenando modelo GPT-2...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(BASE_LOGIC_FILE),
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=str(TRAINED_MODEL_PATH),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        prediction_loss_only=True,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained(str(TRAINED_MODEL_PATH))
    tokenizer.save_pretrained(str(TRAINED_MODEL_PATH))
    print("✅ Entrenamiento finalizado y modelo guardado.")

# ================================
# FASE 2: Procesamiento de datos y contexto
# ================================
def dividir_texto(texto, palabras_por_fragmento=250):
    palabras = texto.split()
    return [" ".join(palabras[i:i+palabras_por_fragmento]) for i in range(0, len(palabras), palabras_por_fragmento)]

def generar_embeddings(fragmentos, model):
    return model.encode(fragmentos, show_progress_bar=True)

def construir_indice(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def buscar_fragmentos(pregunta, embedder, index, fragmentos, k=3):
    emb_pregunta = embedder.encode([pregunta])
    _, indices = index.search(np.array(emb_pregunta), k)
    return [fragmentos[i] for i in indices[0]]

def armar_prompt(pregunta, frag_usuario, frag_wiki, max_characters=800):
    contexto_usuario = "\n".join(frag_usuario) if frag_usuario else ''
    contexto_wiki = "\n".join(frag_wiki) if frag_wiki else ''

    # Truncar el contexto combinado si es muy largo
    total_context = f"{contexto_usuario}\n{contexto_wiki}"
    if len(total_context) > max_characters:
        total_context = total_context[:max_characters]

    return f"""
Responde la siguiente pregunta usando la información proporcionada.

Contexto:
{total_context}

Pregunta: {pregunta}
Respuesta:"""



def cargar_modelo_generador():
    return pipeline("text-generation", model=str(TRAINED_MODEL_PATH), tokenizer=str(TRAINED_MODEL_PATH))

# ================================
# FASE 3: Loop principal
# ================================
def main():
    if not BASE_LOGIC_FILE.exists():
        generar_base_desde_wikipedia()

    if not TRAINED_MODEL_PATH.exists():
        finetune_gpt2()

    print("\n🔍 Generando embeddings para recuperación...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with open(BASE_LOGIC_FILE, "r", encoding="utf-8") as f:
        texto = f.read()
    fragmentos = dividir_texto(texto)
    embeddings = generar_embeddings(fragmentos, embedder)
    index = construir_indice(np.array(embeddings))

    # Cargar archivo del usuario (si existe)
    if CUSTOM_FILE.exists():
        print("📄 Cargando archivo del usuario...")
        with open(CUSTOM_FILE, "r", encoding="utf-8") as f:
            texto_usuario = f.read()
        fragmentos_usuario = dividir_texto(texto_usuario)
        embeddings_usuario = generar_embeddings(fragmentos_usuario, embedder)
        index_usuario = construir_indice(np.array(embeddings_usuario))
    else:
        fragmentos_usuario = []
        index_usuario = None

    generator = cargar_modelo_generador()

    while True:
        pregunta = input("\n🔹 Escribe tu pregunta (o 'exit' para salir): ")
        if pregunta.lower() == "exit":
            break

        frag_usuario = buscar_fragmentos(pregunta, embedder, index_usuario, fragmentos_usuario, k=2) if index_usuario else []
        frag_wiki = buscar_fragmentos(pregunta, embedder, index, fragmentos, k=2)

        prompt = armar_prompt(pregunta, frag_usuario, frag_wiki)

        print("\n🧾 Prompt generado:")
        print(prompt)

        print("\n⏳ Generando respuesta...")
        start = time.time()
        output = generator(prompt, max_new_tokens=100, truncation=True)[0]["generated_text"]
        end = time.time()

        respuesta_final = output.replace(prompt, "").strip()
        print(f"\n🔎 Respuesta generada:\n{respuesta_final}")
        print(f"🕒 Tiempo de respuesta: {end - start:.2f} segundos")

if __name__ == "__main__":
    main()
