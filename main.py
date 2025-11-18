from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


app = FastAPI()


class InputLeituras(BaseModel):
    leituras: list 

modelo = joblib.load("modelo.pkl")


def extrair_features(leituras_json):
    df = pd.DataFrame(leituras_json)

    
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("Cada leitura deve conter 'x', 'y' e 'z'.")

    
    df["accel_magnitude"] = np.sqrt(
        df["x"]**2 + df["y"]**2 + df["z"]**2
    )

    
    window = 10
    accel_mean = df["accel_magnitude"].rolling(window).mean().iloc[-1]
    accel_std  = df["accel_magnitude"].rolling(window).std().iloc[-1]
    accel_max  = df["accel_magnitude"].rolling(window).max().iloc[-1]
    accel_min  = df["accel_magnitude"].rolling(window).min().iloc[-1]

    return [accel_mean, accel_std, accel_max, accel_min]

@app.get("/")
def home():
    return {"message": "API Decision Tree rodando!"}


@app.post("/predict")
def predict(data: InputLeituras):

    # Validação da janela
    if len(data.leituras) < 10:
        return {"error": "Envie exatamente 10 leituras (x, y, z)"}

    try:
        
        features = extrair_features(data.leituras)

       

        arr = np.array(features).reshape(1, -1)

        pred = modelo.predict(arr)

        return {
            "prediction": int(pred[0]),
            "features": {
                "accel_mean": features[0],
                "accel_std": features[1],
                "accel_max": features[2],
                "accel_min": features[3]
            }
        }

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
