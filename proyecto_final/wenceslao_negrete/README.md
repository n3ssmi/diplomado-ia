# **🚀 Reunir Mascotas Perdidas con sus Dueños**

Este repositorio contiene un sistema de IA que ayuda a reunir mascotas perdidas con sus dueños usando técnicas avanzadas de visión por computadora y machine learning.

### **1️⃣ ¿Cómo Funcionaría?**

📸 **Entrada:** El usuario sube una foto de su mascota perdida.

🔍 **Búsqueda:** La IA compara la imagen con una base de datos de mascotas encontradas.

✅ **Salida:** Muestra las coincidencias con un puntaje de similitud.

---

### **2️⃣ Modelos de IA a Utilizar**

🔹 **Clasificación con CNNs (ResNet)** → Identifica raza, color y tamaño.

🔹 **Embeddings con Siamese Networks** → Compara características visuales.

🔹 **GANs??** → Generar variantes de la imagen para mejorar el reconocimiento en distintos ángulos o condiciones.

---

### **3️⃣ Dataset Necesario**

- **🐶 Dataset de Imágenes de Mascotas en México** _(Ej: Fotos de adopción de refugios, sitios de rescate, redes sociales)._
- **📍 Metadatos** _(Raza, color, ubicación y fecha de reporte)._

---

### **4️⃣ Implementación Técnica**

🔹 **Paso 1:** Entrenar un modelo de clasificación para detectar la raza y características generales.

🔹 **Paso 2:** Usar un modelo Siamese para medir la similitud entre mascotas perdidas y encontradas.

🔹 **Paso 3:** Crear una API con **Flask/FastAPI** para consultas en tiempo real.

🔹 **Paso 4:** Implementar en **una app web.**

---

### **5️⃣ Aplicaciones y Beneficios**

✅ **Plataforma colaborativa**: Usuarios pueden reportar mascotas perdidas/encontradas.

✅ **Optimización con IA**: Encuentra coincidencias basadas en similitudes reales, optimizando las búsquedas ineficientes en redes sociales.

✅ **Impacto social**: Facilita la reunificación de mascotas con sus dueños en México.

## Notebook de Desarrollo (pet_finder_poc.ipynb)

El notebook principal implementa un sistema de aprendizaje automático que utiliza similitud de imágenes para identificar posibles coincidencias para mascotas perdidas. Demuestra cómo se extraen características visuales de imágenes de mascotas y cómo implementar búsqueda de similitud para encontrar mascotas en una base de datos.

### Detalles de Implementación

El notebook está organizado en las siguientes secciones:

1. **Preparación de Datos**

   - Utiliza el dataset Oxford-IIIT Pet que contiene imágenes de 37 razas de mascotas
   - Carga opcional de datos a través de TensorFlow datasets

2. **Extracción de Características**

   - Implementa la clase `PetFeatureExtractor` usando varios modelos pre-entrenados:
     - ResNet50, VGG16, MobileNetV2, EfficientNetB3
   - Extrae características profundas de imágenes de mascotas

3. **Base de Datos de Mascotas**

   - Implementa una base de datos en memoria usando DataFrames de pandas
   - Almacena metadatos de mascotas y vectores de características
   - Implementa búsqueda de similitud usando similitud del coseno

4. **Visualización del Espacio de Características**

   - Utiliza t-SNE para visualizar el espacio de características
   - Muestra agrupamiento por especies y razas

5. **Extracción Mejorada de Características**

   - Extiende el extractor básico con histogramas de color usando el espacio de color HSV
   - Añade características de textura con Patrones Binarios Locales
   - Demuestra precisión mejorada en las coincidencias

6. **Búsqueda Mejorada con Metadatos**
   - Implementa similitud ponderada que combina características visuales con metadatos
   - Mejora los resultados de búsqueda considerando ubicación, especie y raza

### Tecnologías Principales

- TensorFlow para modelos de aprendizaje profundo
- scikit-learn para cálculos de similitud y reducción de dimensionalidad
- OpenCV y scikit-image para procesamiento de imágenes
- pandas para manejo de datos

## Requisitos

- Python 3.7+
- TensorFlow 2.4+
- scikit-learn ≥ 0.24.0
- scikit-image ≥ 0.17.2
- OpenCV ≥ 4.5.0
- pandas ≥ 1.1.0
- matplotlib ≥ 3.3.0
- numpy ≥ 1.19.0

## Próximos Pasos

### Mejoras Técnicas

- Implementar Siamese Networks para comparación de imágenes más precisa
- Integrar detección automática de mascotas en imágenes con fondos complejos
- Desarrollar un sistema de recomendación que combine características visuales con metadatos geográficos
- Explorar GANs para generar variaciones de imágenes que mejoren el reconocimiento

### Desarrollo de Plataforma

- Crear una API RESTful con FastAPI para integración con aplicaciones frontend
- Diseñar una interfaz web amigable para reportes y búsquedas
- Implementar sistema de notificaciones para alertar sobre posibles coincidencias

### Expansión de Datos

- Colaborar con refugios de animales y veterinarias para ampliar la base de datos
- Implementar un sistema de crowdsourcing para recolectar imágenes de mascotas
- Crear una estructura de etiquetado colaborativo para mejorar la calidad de los metadatos
