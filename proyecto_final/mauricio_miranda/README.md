# Instalar dependencias

Para instalar las dependencias utiliza el comando:
```bash
pip install -r requirements.txt
```
El codigo que se utilizo para entrenar el modelo esta en la carpeta src llamado **train_llm.py**

Para descargar los pesos del modelo se hace del siguiente enlance:
https://drive.google.com/drive/folders/1vVVZ9DZxHNwqQBv-F3LT4lyIQ8W4dZB0?usp=sharing

# API del modelo con FastAPI

Para ejecutar la api muevete a la carpeta src/api y ejecuta el comando:
```bash
uvicorn main:app
```
