from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import keras_nlp

app = FastAPI()

#Estructura de datos
class PromptInput(BaseModel):
    prompt: str

#Cargar el modelo

model_path = "../../data/model/"
model = gemma_lm = keras_nlp.models.Gemma3CausalLM.from_preset(model_path)
max_sequence_length = 512

@app.get("/")
async def root():
    return {"response":"Hello World!"}

@app.post("/api/chat-bot")
async def process_prompt(data: PromptInput):
    if len(data.prompt.strip()) == 0:
        return JSONResponse(
            status_code = 400,
            content={"response":"El prompt no puede ser vacio."}
        )
    try:
        output = model.generate(inputs=input, max_length=max_sequence_length, strip_prompt=True)
        return JSONResponse(
            status_code = 200,
            content={"response": output}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"response": "Ocurrio un problema al generar la respuesta"}
        )
    
