"""
Bus Overcrowding Prediction System
FastAPI Backend — 10 Real BMTC Routes
Reva University Mini Project | ISE Dept.
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="BMTC CrowdSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "model_artifacts"
model    = pickle.load(open(f"{MODEL_DIR}/bus_model.pkl",  "rb"))
le_route = pickle.load(open(f"{MODEL_DIR}/le_route.pkl",   "rb"))
le_stop  = pickle.load(open(f"{MODEL_DIR}/le_stop.pkl",    "rb"))
le_label = pickle.load(open(f"{MODEL_DIR}/le_label.pkl",   "rb"))
scaler   = pickle.load(open(f"{MODEL_DIR}/scaler.pkl",     "rb"))
features = pickle.load(open(f"{MODEL_DIR}/features.pkl",   "rb"))

STOP_POPULARITY = {
    "Majestic (KBS)": 3.0,
    "Central Silk Board (Hosur RD)": 2.8,
    "Central Silk Board (BTM)": 2.8,
    "Central Silk Board (ORR)": 2.7,
    "Hebbala (Canara Bank)": 2.5,
    "Marathahalli Bridge (ORR)": 2.3,
    "K. R. Puram Rlwy Station": 2.2,
    "Manyata Tech Park": 2.0,
    "ITPL": 2.0,
    "Electronic City (Hosur Rd)": 1.9,
    "Electronic City": 1.9,
    "Indiranagar 100 FT RD (6th Mai)": 1.8,
    "Banashankari Bus Station": 1.7,
    "K. R. Market": 1.6,
    "Koramangala (SONY World)": 1.5,
    "Koramangala 1st Block": 1.5,
    "HSR Layout": 1.4,
    "BTM Check Post": 1.3,
    "Bellandur": 1.3,
    "Kadubeesanahalli": 1.2,
    "Rajajinagar/Navaranga": 1.2,
    "R M C (Yeshwanthpura)": 1.2,
    "Kundalahalli": 1.1,
    "Domlur": 1.1,
    "Jayanagar Water Tank": 1.0,
    "South End Circle": 1.0,
}
DEFAULT_STOP_WEIGHT = 0.7

ROUTE_STOPS = {
    "500D": {
        "name": "Hebbal to Silk Board",
        "stops": [
            "Hebbala (Canara Bank)", "Manyata Tech Park", "Veerannapalya",
            "Kalyan Nagar", "C V Raman Nagar", "Tin Factory",
            "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)",
            "Kadubeesanahalli", "Bellandur", "Iblur",
            "Central Silk Board (BTM)", "Central Silk Board (Hosur RD)"
        ],
    },
    "335E": {
        "name": "Majestic to Whitefield",
        "stops": [
            "Majestic (KBS)", "Corporation", "Domlur",
            "Indiranagar 100 FT RD (6th Mai)", "Marathahalli (Jn. Vartur and O)",
            "Kundalahalli", "ITPL", "BSNL Office (Whitefield RD)"
        ],
    },
    "401": {
        "name": "Kempegowda Bus Station to Yelahanka",
        "stops": [
            "Majestic (KBS)", "Mehkri Circle", "Hebbala (Canara Bank)",
            "Kogilu Cross", "Yelahanka Old Town", "Yelahanka"
        ],
    },
    "250D": {
        "name": "Banashankari to Rajajinagar",
        "stops": [
            "Banashankari Bus Station", "Jayanagar Water Tank",
            "South End Circle", "Lalbagh West Gate",
            "K. R. Market", "City Market", "Rajajinagar/Navaranga"
        ],
    },
    "G5": {
        "name": "Kempegowda Bus Station to Sarjapur",
        "stops": [
            "Majestic (KBS)", "K. R. Market", "Lalbagh West Gate",
            "Koramangala 1st Block", "Koramangala (SONY World)",
            "Ejipura", "Kaggadasapura Jn (Big Bazaar)",
            "Sarjapur Road", "Sarjapur"
        ],
    },
    "201R": {
        "name": "Hebbal to Electronic City",
        "stops": [
            "Hebbala (Canara Bank)", "Manyata Tech Park",
            "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)",
            "Central Silk Board (ORR)", "Electronic City (Hosur Rd)",
            "Electronic City"
        ],
    },
    "356F": {
        "name": "Yeshwanthpur to BTM Layout",
        "stops": [
            "R M C (Yeshwanthpura)", "Rajajinagar/Navaranga",
            "Bhashyam Circle (Rajajinagar)", "Majestic (KBS)",
            "South End Circle", "BTM Check Post",
            "Kuvempu Nagar (BTM Layout)", "HSR Layout"
        ],
    },
    "600K": {
        "name": "Koramangala to Indiranagar",
        "stops": [
            "Koramangala (80 FT RD)", "Koramangala (SONY World)",
            "Ejipura", "Domlur",
            "Indiranagar 100 FT RD (6th Mai)",
            "Indiranagar 100 FT RD (13th Ma)", "Binnamangala"
        ],
    },
    "V1": {
        "name": "Banashankari to Marathahalli",
        "stops": [
            "Banashankari Bus Station", "Jayanagar Water Tank",
            "Central Silk Board (BTM)", "HSR Layout (BDA Complex)",
            "HSR Layout", "Bellandur",
            "Marathahalli (Jn. Vartur and O)", "Marathahalli Bridge (ORR)"
        ],
    },
    "C9": {
        "name": "Majestic to HSR Layout",
        "stops": [
            "Majestic (KBS)", "Lalbagh West Gate", "K. R. Market",
            "Koramangala 1st Block", "Koramangala (80 FT RD)",
            "Central Silk Board (BTM)", "HSR Layout (BDA Complex)",
            "HSR (Depot-25)"
        ],
    },
}


class PredictRequest(BaseModel):
    route_id: str
    stop_name: str
    hour: int
    day_of_week: int


@app.get("/")
def root():
    return {"message": "BMTC CrowdSense API running!", "routes": len(ROUTE_STOPS)}


@app.get("/routes")
def get_routes():
    return {"routes": {k: v["name"] for k, v in ROUTE_STOPS.items()}}


@app.get("/stops/{route_id}")
def get_stops(route_id: str):
    if route_id not in ROUTE_STOPS:
        return {"error": "Route not found", "stops": []}
    return {
        "route_id": route_id,
        "name": ROUTE_STOPS[route_id]["name"],
        "stops": ROUTE_STOPS[route_id]["stops"]
    }


@app.post("/predict")
def predict(req: PredictRequest):
    is_weekend  = int(req.day_of_week >= 5)
    route_data  = ROUTE_STOPS.get(req.route_id, {})
    stops       = route_data.get("stops", [])
    stop_index  = stops.index(req.stop_name) if req.stop_name in stops else 0
    total_stops = len(stops)
    popularity  = STOP_POPULARITY.get(req.stop_name, DEFAULT_STOP_WEIGHT)

    hour_sin = np.sin(2 * np.pi * req.hour / 24)
    hour_cos = np.cos(2 * np.pi * req.hour / 24)
    dow_sin  = np.sin(2 * np.pi * req.day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * req.day_of_week / 7)

    route_enc = le_route.transform([req.route_id])[0] if req.route_id in le_route.classes_ else 0
    stop_enc  = le_stop.transform([req.stop_name])[0] if req.stop_name in le_stop.classes_ else 0
    scale_vals = scaler.transform([[req.hour, req.day_of_week, stop_index, total_stops, popularity]])[0]

    row = {
        "route_id_enc":  route_enc,
        "stop_name_enc": stop_enc,
        "stop_index":    scale_vals[2],
        "total_stops":   scale_vals[3],
        "hour":          scale_vals[0],
        "day_of_week":   scale_vals[1],
        "is_weekend":    is_weekend,
        "hour_sin":      hour_sin,
        "hour_cos":      hour_cos,
        "dow_sin":       dow_sin,
        "dow_cos":       dow_cos,
        "stop_popularity": scale_vals[4],
    }
    X = pd.DataFrame([row])[features]
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_label = le_label.inverse_transform([pred_enc])[0]

    return {
        "label":    pred_label,
        "route_id": req.route_id,
        "stop_name": req.stop_name,
        "confidence": {
            str(le_label.classes_[i]): round(float(pred_proba[i]) * 100, 1)
            for i in range(len(pred_proba))
        }
    }