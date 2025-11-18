from fastapi import FastAPI
import uvicorn
import firebase_admin
from firebase_admin import credentials, db
import joblib
import threading
import time
import numpy as np
from pydantic import BaseModel
import os

# -----------------------------------------------------------
# Iniciar API
# -----------------------------------------------------------
app = FastAPI()

class InputData(BaseModel):
    features : list

@app.get("/")
def root():
    return {"message": "API Decision Tree rodando!"}

# -----------------------------------------------------------
# Inicializar Firebase
# -----------------------------------------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://projetotcc-b0798-default-rtdb.firebaseio.com/"
})

# -----------------------------------------------------------
# Carregar Modelo IA
# -----------------------------------------------------------
modelo = joblib.load("modelo.pkl")  # Seu arquivo do modelo

# -----------------------------------------------------------
# Loop de processamento automático
# -----------------------------------------------------------
def loop_firebase():
    while True:
        try:
            # pegar dados brutos
            ref = db.reference("/dados_brutos")
            dados = ref.get()

            if dados:
                print("Dados recebidos:", dados)

                # modelo espera lista - manter como lista
                pred = modelo.predict([dados])[0]

                print("Resultado:", pred)

                # gravar resultado
                db.reference("/dados_processados").set({
                    "resultado": float(pred),
                    "timestamp": time.time()
                })

                # limpar dados brutos
                ref.delete()

        except Exception as e:
            print("Erro no loop:", e)

        time.sleep(5)  # repete a cada 5 segundos

# rodar thread
threading.Thread(target=loop_firebase, daemon=True).start()

# -----------------------------------------------------------
# Iniciar servidor FASTAPI
# -----------------------------------------------------------

@app.post("/predict")
def predict(data : InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = modelo.predict(arr)
    return {"prediction": int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
