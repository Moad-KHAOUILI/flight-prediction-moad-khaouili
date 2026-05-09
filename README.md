# Flight Delay and Overbooking Prediction

Two binary classification models trained on a merged dataset of US domestic flight records:
1. Predict whether a flight will be delayed by 15 or more minutes
2. Predict whether a flight will be overbooked

---

## Datasets

| File | Description |
|---|---|
| `full_data_flightdelay.csv` | ~6.5M US domestic flights from 2019 (BTS). Sampled to last 10,000 rows. |
| `Air_Traffic_Passenger_Statistics.csv` | Monthly passenger counts at SFO by airline (public). |
| `flight_bookings.csv` | Synthetic per-flight booking records generated from the SFO data. |
| `merged_flights.csv` | Final training dataset: delay sample joined with booking stats on airline + month. |
| `ASRS_DBOnline.csv` | ASRS incident reports (optional LLM risk score feature). Airports are anonymized. |


## Models

Both models use a Multi-Layer Perceptron (3 hidden layers: 128, 64, 32 neurons, ReLU, Dropout).
Categorical features are one-hot encoded. Numerical features are standard scaled.
Class weights are applied to handle imbalance. Training uses early stopping.


## Repository Structure

```
data/                        Raw and processed datasets
notebooks/
    flight_predictor.ipynb   Main notebook with all steps
scripts/
    gen_nb.py                Script used to generate the notebook
TECHNICAL_REPORT.md
README.md
```


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn torch transformers matplotlib
```
