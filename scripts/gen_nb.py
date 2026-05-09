import json, uuid

def uid(): return uuid.uuid4().hex[:8]
def md(s): return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "id": uid(), "metadata": {}, "source": s, "outputs": [], "execution_count": None}

cells = []

cells.append(md(
    "# Flight Delay and Overbooking Prediction\n"
    "**Course:** Advanced AI -- 2025/2026  \n"
    "**Author:** Moad Khaouili\n\n"
    "Two binary classification models:\n"
    "1. **Flight Delay Prediction** -- will a flight depart more than 15 minutes late?\n"
    "2. **Overbooking Prediction** -- is a flight overbooked?\n\n---"
))

cells.append(md(
    "## Section 1 -- Data Preparation\n\n"
    "> **All data files have already been prepared and saved to `data/`. "
    "This section explains what was done -- no need to re-run.**\n\n---\n\n"
    "Two datasets were combined to build the training data:\n\n"
    "- **Flight delay data** (`full_data_flightdelay.csv`): ~6.5M US domestic flights from 2019. "
    "The last 10,000 rows were extracted as `sample_10k_flightdelay.csv`. "
    "Contains the delay target (`DEP_DEL15`) and flight/weather features.\n"
    "- **Synthetic booking data** (`flight_bookings.csv`): ~366k simulated flight records "
    "generated from public SFO passenger statistics, containing seat counts, passenger counts, "
    "load factors, and an overbooking label.\n\n"
    "**These two datasets were merged on: airline name + month.**\n\n"
    "Before joining, carrier names were normalized -- lowercased and stripped of punctuation "
    "and legal suffixes such as Inc., Co., Ltd. -- so names like `Delta Air Lines Inc.` and "
    "`Delta Air Lines` would match correctly. The bookings data was then aggregated per "
    "(airline, month), computing average load factor, average passengers, and overbooking rate, "
    "and left-joined onto the delay sample.\n\n"
    "81.4% of rows found a match. The remaining 18.6% (carriers not in the SFO dataset, "
    "e.g. United, Spirit) have NaN for booking columns and will be imputed before training.\n\n"
    "**Output:** `merged_flights.csv` -- 10,000 rows x 30 columns."
))

cells.append(md(
    "## Section 2 -- Load Dataset\n\n"
    "Load the pre-built merged dataset and verify its contents before feature engineering."
))

cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "df = pd.read_csv(\"../data/merged_flights.csv\")\n"
    "print(f\"Shape: {df.shape}\")\n"
    "print(f\"Columns: {df.columns.tolist()}\")\n"
    "df.head()"
))

cells.append(code(
    "print(\"=== Dataset Summary ===\")\n"
    "print(f\"Total rows: {len(df):,}\")\n"
    "print(f\"Total columns: {df.shape[1]}\")\n"
    "print(\"\\n--- Delay Target (DEP_DEL15) ---\")\n"
    "print(df[\"DEP_DEL15\"].value_counts(normalize=True).round(3))\n"
    "print(\"\\n--- Booking Feature Coverage ---\")\n"
    "booking_cols = [\"avg_load_factor\", \"avg_passengers\", \"overbooked_rate\", \"avg_seats_booked\"]\n"
    "print(df[booking_cols].notna().mean().round(3))\n"
    "print(\"\\n--- Null counts (columns with nulls only) ---\")\n"
    "nulls = df.isnull().sum()\n"
    "print(nulls[nulls > 0])"
))

cells.append(md(
    "## Section 3 -- ASRS LLM Risk Score Feature\n\n"
    "This section adds an optional feature derived from the **Aviation Safety Reporting System (ASRS)** "
    "dataset. ASRS collects voluntary incident reports submitted by pilots, controllers, and crew -- "
    "each containing a free-text narrative describing a safety-related event.\n\n"
    "### Concept\n\n"
    "Each incident narrative is passed through an LLM (zero-shot classifier) which returns a "
    "**risk score between 0 and 1** reflecting how likely the described event is to contribute "
    "to a flight delay. Scores are aggregated by airport and month, then merged into the main dataset. "
    "Flights with no matching ASRS report are assigned a score of **0** (no incident reported).\n\n"
    "### Why this helps\n\n"
    "Standard features like weather or congestion capture systemic delay causes. ASRS narratives "
    "capture rare but high-impact events -- equipment failures, near-misses, ATC issues -- that "
    "would otherwise be invisible to the model. The LLM converts unstructured text into a single "
    "interpretable numeric feature.\n\n"
    "### Merge key\n\n"
    "The ASRS score is joined onto the delay dataset using:\n"
    "- `DEPARTING_AIRPORT` -- departure airport name\n"
    "- `MONTH` -- month of the flight (1-12)\n\n"
    "When multiple reports exist for the same airport+month, their scores are averaged.\n\n"
    "### Pipeline\n\n"
    "```\n"
    "ASRS narratives (free text)\n"
    "        |\n"
    "        v\n"
    "LLM zero-shot classifier\n"
    "        |\n"
    "        v\n"
    "risk_score per report (0 to 1)\n"
    "        |\n"
    "        v\n"
    "Aggregate: mean score per (airport, month)\n"
    "        |\n"
    "        v\n"
    "Left join onto merged_flights.csv\n"
    "        |\n"
    "        v\n"
    "Fill missing with 0  (no incident reported)\n"
    "```"
))

cells.append(code(
    "from transformers import pipeline\n\n"
    "# Load zero-shot classification model as the LLM risk scorer\n"
    "scorer = pipeline(\n"
    "    \"zero-shot-classification\",\n"
    "    model=\"facebook/bart-large-mnli\"\n"
    ")\n\n"
    "CANDIDATE_LABELS = [\n"
    "    \"likely to cause a flight delay\",\n"
    "    \"unlikely to cause a flight delay\"\n"
    "]\n\n"
    "def score_narrative(text):\n"
    "    \"\"\"Return delay risk score (0-1) for a single ASRS narrative.\"\"\"\n"
    "    if not isinstance(text, str) or len(text.strip()) == 0:\n"
    "        return 0.0\n"
    "    result = scorer(text[:512], candidate_labels=CANDIDATE_LABELS)\n"
    "    return round(result[\"scores\"][0], 4)\n\n"
    "print(\"LLM scorer loaded.\")"
))

cells.append(md(
    "### Load and Score ASRS Reports\n\n"
    "Expected columns in the ASRS CSV:\n\n"
    "| Column | Description |\n"
    "|---|---|\n"
    "| `narrative` | Free-text incident report |\n"
    "| `airport` | Airport name matching `DEPARTING_AIRPORT` |\n"
    "| `month` | Month of the incident (1-12) |\n\n"
    "A small demo sample is used below to illustrate the pipeline."
))

cells.append(code(
    "# To use real ASRS data, replace this with:\n"
    "# df_asrs = pd.read_csv(\"../data/asrs_reports.csv\")\n\n"
    "# --- Demo sample ---\n"
    "df_asrs = pd.DataFrame({\n"
    "    \"narrative\": [\n"
    "        \"Aircraft experienced hydraulic failure on taxiway, causing significant ground delay.\",\n"
    "        \"Routine flight, no issues reported.\",\n"
    "        \"ATC miscommunication led to runway incursion, departure held for 40 minutes.\",\n"
    "        \"Bird strike on approach, aircraft returned to gate for inspection.\",\n"
    "        \"Normal operations, weather clear.\",\n"
    "    ],\n"
    "    \"airport\": [\n"
    "        \"McCarran International\",\n"
    "        \"Orlando International\",\n"
    "        \"McCarran International\",\n"
    "        \"Boise Air Terminal\",\n"
    "        \"Orlando International\",\n"
    "    ],\n"
    "    \"month\": [1, 1, 3, 2, 2]\n"
    "})\n\n"
    "df_asrs[\"risk_score\"] = df_asrs[\"narrative\"].apply(score_narrative)\n"
    "df_asrs[[\"airport\", \"month\", \"risk_score\", \"narrative\"]]"
))

cells.append(code(
    "# Aggregate: mean risk score per (airport, month)\n"
    "asrs_agg = df_asrs.groupby([\"airport\", \"month\"])[\"risk_score\"].mean().reset_index()\n"
    "asrs_agg.columns = [\"DEPARTING_AIRPORT\", \"MONTH\", \"asrs_risk_score\"]\n"
    "asrs_agg[\"asrs_risk_score\"] = asrs_agg[\"asrs_risk_score\"].round(4)\n\n"
    "# Left join onto main dataset\n"
    "df = df.merge(asrs_agg, on=[\"DEPARTING_AIRPORT\", \"MONTH\"], how=\"left\")\n\n"
    "# Flights with no ASRS report get score 0\n"
    "df[\"asrs_risk_score\"] = df[\"asrs_risk_score\"].fillna(0.0)\n\n"
    "print(f\"asrs_risk_score added.\")\n"
    "print(f\"Non-zero entries: {(df['asrs_risk_score'] > 0).sum()} / {len(df)}\")\n"
    "print(df[\"asrs_risk_score\"].describe().round(4))"
))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11.0"}},
      "cells": cells}
with open("notebooks/flight_predictor.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Done --", len(cells), "cells")
