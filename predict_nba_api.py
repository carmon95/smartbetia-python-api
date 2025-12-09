import os
import requests
import joblib

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


# ------------------------------
# Configuración del modelo local
# ------------------------------
MODEL_PATH = "nba_over_model.joblib"

app = FastAPI(
    title="SmartBet IA - NBA Over Model",
    version="1.0.0"
)

# Cargar modelo al iniciar
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# ------------------------------
# Configuración para el JOB (Supabase + API de cuotas)
# ------------------------------
# Token simple para proteger el endpoint del job
JOB_TOKEN = os.getenv("JOB_TOKEN", "cambia-este-token")

# Datos de Supabase (Proyecto donde está la tabla value_bets)
# Ejemplo SUPABASE_URL: https://tu-proyecto.supabase.co
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Datos de la API de cuotas (TheOddsAPI en tu caso)
ODDS_API_URL = os.getenv("ODDS_API_URL")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Mínimo edge para considerar una apuesta como value bet
# edge = prob_over * odds - 1
# Si MIN_EDGE = 0.05, se guardan solo bets con >= 5% de edge.
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.0"))


# ------------------------------
# Esquemas de petición / respuesta del modelo
# ------------------------------
class NbaOverRequest(BaseModel):
    closing_total_line: float
    over_odds: float


class NbaOverResponse(BaseModel):
    prob_over: float


# ------------------------------
# Endpoint principal: predicción de Over
# ------------------------------
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


# ------------------------------
# Helpers para Supabase y API de cuotas
# ------------------------------
def insert_value_bet(row: dict):
    """
    Inserta una fila en la tabla value_bets de Supabase usando la REST API.
    'row' debe contener las columnas que existen en tu tabla value_bets.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL o SUPABASE_ANON_KEY no están configurados")

    url = f"{SUPABASE_URL}/rest/v1/value_bets"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    resp = requests.post(url, json=row, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_real_odds():
    """
    Llama a TheOddsAPI (ODDS_API_URL) para traer partidos y cuotas reales NBA Totals.

    Estructura esperada de cada evento (simplificada):

    {
      "commence_time": "2025-12-09T23:10:00Z",
      "home_team": "Orlando Magic",
      "away_team": "Miami Heat",
      "bookmakers": [
        {
          "markets": [
            {
              "key": "totals",
              "outcomes": [
                { "name": "Over", "price": 1.90, "point": 224.5 },
                { "name": "Under", "price": 1.90, "point": 224.5 }
              ]
            }
          ]
        }
      ]
    }
    """
    if not ODDS_API_URL or not ODDS_API_KEY:
        # Devolvemos lista vacía para no romper el job si aún no configuras la API
        return []

    resp = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": ODDS_API_KEY,   # TheOddsAPI usa apiKey así
            "regions": "us",
            "markets": "totals",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def compute_edge(prob_over: float, over_odds: float) -> float:
    """
    edge = prob_over * over_odds - 1
    Si el resultado es, por ejemplo, 0.05, eso es un +5% de edge.
    """
    return prob_over * over_odds - 1.0


# ------------------------------
# Endpoint de JOB: actualizar value_bets automáticamente
# ------------------------------
@app.post("/job/update-value-bets")
async def job_update_value_bets(request: Request):
    """
    Endpoint que será llamado automáticamente (por ejemplo, desde un CRON en Supabase)
    para:
      1) Obtener partidos y cuotas reales desde TheOddsAPI.
      2) Calcular prob_over y edge usando tu modelo.
      3) Insertar value bets en la tabla value_bets de Supabase.

    NO es para que lo llame la app Android; solo lo usarán jobs programados.
    """
    # 1) Validar el token
    token = request.query_params.get("token")
    if token != JOB_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2) Verificar configuración mínima
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL o SUPABASE_ANON_KEY no están configurados en el servidor",
        )

    odds_data = fetch_real_odds()
    if not odds_data:
        return {
            "status": "ok",
            "inserted_count": 0,
            "message": "Job ejecutado, pero ODDS_API_URL/ODDS_API_KEY no están configurados o no devolvieron datos.",
        }

    inserted_count = 0
    total_events = 0
    with_bookmakers = 0
    with_totals = 0
    with_over = 0

    for event in odds_data:
        total_events += 1

        home_team = event.get("home_team")
        away_team = event.get("away_team")
        kickoff_at = event.get("commence_time")  # ISO 8601
        sport = "NBA"
        league = "NBA"

        bookmakers = event.get("bookmakers") or []
        if not bookmakers:
            continue
        with_bookmakers += 1

        # Tomamos el primer bookmaker disponible
        bookmaker = bookmakers[0]
        markets = bookmaker.get("markets") or []

        # Buscamos el mercado "totals"
        totals_market = next((m for m in markets if m.get("key") == "totals"), None)
        if not totals_market:
            continue
        with_totals += 1

        outcomes = totals_market.get("outcomes") or []

        # En TheOddsAPI, Outcomes para totals tienen:
        # { "name": "Over" / "Under", "price": 1.90, "point": 224.5 }
        over_outcome = next(
            (o for o in outcomes if str(o.get("name", "")).lower() == "over"),
            None,
        )
        if not over_outcome:
            continue
        with_over += 1

        try:
            closing_total_line = float(over_outcome["point"])   # línea total
            over_odds = float(over_outcome["price"])            # cuota
        except Exception:
            # Si viene algo raro en el formato, saltamos este evento
            continue

        # 3) Calcular prob_over con TU modelo (misma lógica que en /predict-nba-over)
        implied_over_prob = 1.0 / over_odds
        X = [[closing_total_line, implied_over_prob]]
        prob_over = float(model.predict_proba(X)[0, 1])

        # 4) Calcular edge (solo informativo por ahora)
        edge = compute_edge(prob_over, over_odds)

        # ⚠️ MODO DEBUG: NO filtramos por edge
        # if edge < MIN_EDGE:
        #     continue

        # 5) Armar fila para la tabla value_bets
        row = {
            "sport": sport,
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "market": "TOTAL_POINTS_OVER",
            "closing_total_line": closing_total_line,
            "odds": over_odds,
            "edge": edge,
            "kickoff_at": kickoff_at,
            "is_active": True,
        }

        # ⚠️ MODO DEBUG: si hay error insertando, queremos que reviente
        insert_value_bet(row)
        inserted_count += 1

    return {
        "status": "ok",
        "inserted_count": inserted_count,
        "total_events": total_events,
        "with_bookmakers": with_bookmakers,
        "with_totals": with_totals,
        "with_over": with_over,
    }