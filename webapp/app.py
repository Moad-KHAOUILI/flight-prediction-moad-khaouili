import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template

# ── MLP definition (must match training) ───────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ── Load artifacts ──────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "models")

with open(os.path.join(BASE, "meta.json")) as f:
    meta = json.load(f)

with open(os.path.join(BASE, "scaler_delay.pkl"), "rb") as f:
    scaler_delay = pickle.load(f)

with open(os.path.join(BASE, "scaler_ob.pkl"), "rb") as f:
    scaler_ob = pickle.load(f)

model_delay = MLP(meta["delay_input_dim"])
model_delay.load_state_dict(torch.load(os.path.join(BASE, "model_delay.pt"), map_location="cpu"))
model_delay.eval()

model_ob = MLP(meta["ob_input_dim"])
model_ob.load_state_dict(torch.load(os.path.join(BASE, "model_ob.pt"), map_location="cpu"))
model_ob.eval()

FEATURE_COLS_DELAY = meta["feature_cols_delay"]
FEATURE_COLS_OB    = meta["feature_cols_ob"]

# ── Flask app ───────────────────────────────────────────────────────────────

_nlp_scorer = None
CANDIDATE_LABELS = [
    'likely to cause a flight delay',
    'unlikely to cause a flight delay',
]

def get_scorer():
    global _nlp_scorer
    if _nlp_scorer is None:
        from transformers import pipeline
        _nlp_scorer = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    return _nlp_scorer

def score_narrative(text):
    text = str(text).strip()
    if not text:
        return 0.0
    result = get_scorer()(text[:512], candidate_labels=CANDIDATE_LABELS)
    return round(float(result['scores'][0]), 4)

app = Flask(__name__)

def _predict(model, scaler, feature_cols, raw_input: dict):
    """Build feature vector, scale, run model, return probability."""
    vec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in raw_input:
            vec[i] = float(raw_input[col])
    vec = scaler.transform(vec.reshape(1, -1))
    with torch.no_grad():
        logit = model(torch.tensor(vec, dtype=torch.float32))
        prob  = torch.sigmoid(logit).item()
    return round(prob, 4)


@app.route("/")
def index():
    carriers = [
        "Alaska Airlines Inc.", "Allegiant Air", "American Airlines Inc.",
        "American Eagle Airlines Inc.", "Atlantic Southeast Airlines",
        "Comair Inc.", "Delta Air Lines Inc.", "Endeavor Air Inc.",
        "Frontier Airlines Inc.", "Hawaiian Airlines Inc.", "JetBlue Airways",
        "Mesa Airlines Inc.", "Midwest Airline, Inc.", "SkyWest Airlines Inc.",
        "Southwest Airlines Co.", "Spirit Air Lines", "United Air Lines Inc.",
    ]
    airports = [
        "Atlanta Municipal", "Austin - Bergstrom International", "Boise Air Terminal",
        "Chicago O'Hare International", "Fort Lauderdale-Hollywood International",
        "General Mitchell Field", "Honolulu International", "Kansas City International",
        "Logan International", "Los Angeles International", "McCarran International",
        "Miami International", "Minneapolis-St Paul International",
        "Nashville International", "Orange County", "Orlando International",
        "Palm Springs International", "Phoenix Sky Harbor International",
        "Portland International", "Raleigh-Durham International",
        "Reno/Tahoe International", "Sacramento International",
        "Salt Lake City International", "San Antonio International",
        "San Diego International Lindbergh Fl", "San Francisco International",
        "Seattle International", "Southwest Florida International",
        "Tampa International", "Tulsa International",
    ]
    dep_blocks = [
        "0001-0559","0600-0659","0700-0759","0800-0859","0900-0959",
        "1000-1059","1100-1159","1200-1259","1300-1359","1400-1459",
        "1500-1559","1600-1659","1700-1759","1800-1859","1900-1959",
        "2000-2059","2100-2159","2200-2259","2300-2359",
    ]
    airlines_ob = sorted([
        "Aer Lingus","Aeromexico","Air Berlin","Air Canada ","Air Canada Jazz",
        "Air China","Air France","Air India Limited","Air New Zealand",
        "AirTran Airways","Alaska Airlines","All Nippon Airways","Allegiant Air",
        "American Airlines","American Eagle Airlines","Ameriflight","Asiana Airlines",
        "Atlantic Southeast Airlines","BelAir Airlines","British Airways",
        "COPA Airlines, Inc.","Cathay Pacific","China Airlines","China Eastern",
        "China Southern","Compass Airlines","Delta Air Lines","EVA Airways",
        "Emirates ","Etihad Airways","Evergreen International Airlines",
        "ExpressJet Airlines","Frontier Airlines","Hawaiian Airlines","Horizon Air ",
        "Icelandair","Independence Air","Japan Airlines","Jet Airways",
        "JetBlue Airways ","KLM Royal Dutch Airlines","Korean Air Lines","LAN Peru",
        "Lufthansa German Airlines","Mesa Airlines","Mesaba Airlines",
        "Mexicana Airlines","Miami Air International","Midwest Airlines",
        "Northwest Airlines","Pacific Aviation","Philippine Airlines","Qantas Airways",
        "Republic Airlines","SAS Airlines","Servisair","Singapore Airlines",
        "SkyWest Airlines","Southwest Airlines","Spirit Airlines","Sun Country Airlines",
        "Swiss International","Swissport USA","TACA","Turkish Airlines","US Airways",
        "United Airlines","United Airlines - Pre 07/01/2013","Virgin America",
        "Virgin Atlantic","WestJet Airlines","World Airways","XL Airways France",
        "Xtra Airways",
    ])
    return render_template("index.html",
                           carriers=carriers,
                           airports=airports,
                           dep_blocks=dep_blocks,
                           airlines_ob=airlines_ob)


@app.route("/api/predict/delay", methods=["POST"])
def predict_delay():
    data = request.get_json(force=True)

    asrs_text = str(data.get('asrs_text', '')).strip()
    asrs_risk_score = score_narrative(asrs_text)
    asrs_scored = bool(asrs_text)

    raw = {}
    for col in [
        "MONTH","DAY_OF_WEEK","DISTANCE_GROUP","SEGMENT_NUMBER",
        "CONCURRENT_FLIGHTS","NUMBER_OF_SEATS",
        "AIRPORT_FLIGHTS_MONTH","AIRLINE_FLIGHTS_MONTH","AIRLINE_AIRPORT_FLIGHTS_MONTH",
        "AVG_MONTHLY_PASS_AIRPORT","AVG_MONTHLY_PASS_AIRLINE",
        "FLT_ATTENDANTS_PER_PASS","GROUND_SERV_PER_PASS",
        "PLANE_AGE","LATITUDE","LONGITUDE","PRCP","SNOW","SNWD","TMAX","AWND",
        "avg_load_factor","avg_passengers","overbooked_rate","avg_seats_booked",
    ]:
        if col in data:
            raw[col] = data[col]

    raw["asrs_risk_score"] = asrs_risk_score

    # One-hot: CARRIER_NAME
    carrier = data.get("CARRIER_NAME", "")
    carrier_col = f"CARRIER_NAME_{carrier}"
    if carrier_col in FEATURE_COLS_DELAY:
        raw[carrier_col] = 1.0

    # One-hot: DEP_TIME_BLK
    blk = data.get("DEP_TIME_BLK", "")
    blk_col = f"DEP_TIME_BLK_{blk}"
    if blk_col in FEATURE_COLS_DELAY:
        raw[blk_col] = 1.0

    # One-hot: DEPARTING_AIRPORT
    apt = data.get("DEPARTING_AIRPORT", "")
    apt_col = f"DEPARTING_AIRPORT_{apt}"
    if apt_col in FEATURE_COLS_DELAY:
        raw[apt_col] = 1.0

    prob = _predict(model_delay, scaler_delay, FEATURE_COLS_DELAY, raw)

    # Post-hoc ASRS blend: the model trained on aggregated monthly scores (low range),
    # so a high individual narrative score is out-of-distribution and under-weighted.
    # Blend the model probability toward 1.0 proportionally when score > 0.55.
    if asrs_scored and asrs_risk_score > 0.55:
        blend = (asrs_risk_score - 0.55) / 0.45  # 0..1 as score goes 0.55..1.0
        prob = round(prob + blend * (1.0 - prob) * 0.75, 4)
        prob = min(prob, 0.99)

    return jsonify({"probability": prob, "delayed": prob >= 0.68, "asrs_risk_score": asrs_risk_score, "asrs_scored": asrs_scored})


@app.route("/api/predict/overbooking", methods=["POST"])
def predict_overbooking():
    data = request.get_json(force=True)
    raw = {}

    # Numeric
    for col in ["month","seats","passengers","load_factor"]:
        if col in data:
            raw[col] = data[col]

    # One-hot: airline
    al = data.get("airline", "")
    al_col = f"airline_{al}"
    if al_col in FEATURE_COLS_OB:
        raw[al_col] = 1.0

    # One-hot: geo_summary
    gs = data.get("geo_summary", "")
    gs_col = f"geo_summary_{gs}"
    if gs_col in FEATURE_COLS_OB:
        raw[gs_col] = 1.0

    # One-hot: geo_region
    gr = data.get("geo_region", "")
    gr_col = f"geo_region_{gr}"
    if gr_col in FEATURE_COLS_OB:
        raw[gr_col] = 1.0

    # One-hot: terminal
    term = data.get("terminal", "")
    term_col = f"terminal_{term}"
    if term_col in FEATURE_COLS_OB:
        raw[term_col] = 1.0

    # One-hot: price_category
    pc = data.get("price_category", "")
    pc_col = f"price_category_{pc}"
    if pc_col in FEATURE_COLS_OB:
        raw[pc_col] = 1.0

    prob = _predict(model_ob, scaler_ob, FEATURE_COLS_OB, raw)
    return jsonify({"probability": prob, "overbooked": prob >= 0.93})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
