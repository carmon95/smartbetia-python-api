from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = "nba_over_model.joblib"

app = FastAPI(
    title="SmartBet IA - NBA Over Model",
    version="1.0.0"
)

# Cargar modelo al iniciar
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


class NbaOverRequest(BaseModel):
    closing_total_line: float
    over_odds: float


class NbaOverResponse(BaseModel):
    prob_over: float


@app.post("/predict-nba-over", response_model=NbaOverResponse)
def predict_nba_over(payload: NbaOverRequest):
    """
    Recibe:
      - closing_total_line: línea de puntos del partido (ej: 224.5)
      - over_odds: cuota decimal del Over (ej: 1.90)

    Devuelve:
      - prob_over: probabilidad estimada de que se cumpla el Over (0.0 - 1.0)
    """
    # Feature engineered igual que en el entrenamiento
    implied_over_prob = 1.0 / payload.over_odds
    X = [[payload.closing_total_line, implied_over_prob]]

    prob_over = float(model.predict_proba(X)[0, 1])

    return NbaOverResponse(prob_over=prob_over)
