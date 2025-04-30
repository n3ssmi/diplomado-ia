#Importamos las librerias necesarias
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
import time
import keras
import keras_nlp

#Configuramos algunas rutas importantes
path_project = "../data/"
file_kaggle = "kaggle.json"
data_files = ["preguntas_diversas.json", "preguntas_tramites.json"]
data_file = "preguntas_respuestas.json"
model_name = "gemma3_instruct_1b"
model_path = "model/"

#Cargar las credenciales desde el archivo proporcionado por kaggle
with open(path_project + file_kaggle, 'r') as file:
  kaggle_credentials = json.load(file)

#Establecer las variables de entorno
os.environ["KAGGLE_USERNAME"] = kaggle_credentials['username']
os.environ["KAGGLE_KEY"] = kaggle_credentials['key']

#Autenticar con la API de Kaggle
api= KaggleApi()
api.authenticate()

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

keras.mixed_precision.set_global_policy('mixed_bfloat16')

#Lista para guardar los objetos de tipo {"query": value, "response":value}
data_raw_training = []

#Cargamos los datos de los archivos
for file_name in data_files:
  with open(path_project + file_name, 'r') as file:
    data = json.load(file)
  for array in data.values():
    for element in array:
      data_raw_training.append(element)

with open(path_project + data_file, 'r') as file:
  data = json.load(file)

for element in data['preguntas-respuestas']:
  data_raw_training.append(element)

total_size = len(data_raw_training)
print(f"Tamaño del dataset: {len(data_raw_training)}")

#Creamos un dataset de tensorflow con la estructura que recibe el modelo
training_dataset = tf.data.Dataset.from_generator(
    lambda: ({"prompts": item['query'], "responses": item['response']} for item in data_raw_training),
    output_signature={
        "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
        "responses": tf.TensorSpec(shape=(), dtype=tf.string),
    }
)

training_dataset = training_dataset.shuffle(1000).batch(batch_size=1).prefetch(tf.data.AUTOTUNE)

#Creamos una instancia del modelo
gemma_lm = keras_nlp.models.Gemma3CausalLM.from_preset(model_name)
gemma_lm.summary()

#configuramos algunos hiperparametros
sequence_length = 2048
learning_rate = 1e-5
epochs = 3
rank = 8

#habilitamos LoRA con rank = 8
gemma_lm.backbone.enable_lora(rank=rank)
gemma_lm.summary()

#Configuramos el modelo para hacer fine tuning con los nuevos datos
gemma_lm.preprocessor.sequence_length = sequence_length

optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01,
)

optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(training_dataset, epochs=epochs)

#Guardamos el modelo
gemma_lm.save_to_preset(path_project + model_path)

#Probamos el modelo
input = "¿Como inicio el proceso de titulacion en licenciatura?"
max_sequence_length = 2048

inicio = time.time()
output = gemma_lm.generate(inputs=input, max_length=max_sequence_length, strip_prompt=True)
final = time.time()
duracion = final - inicio
print(f'Respuesta: {output}\nTiempo transcurrido: {duracion:.2f}')