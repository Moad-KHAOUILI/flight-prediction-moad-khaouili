import json

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

new_source = """\
# The ASRS CSV has two header rows; load with header=[0,1] to get MultiIndex columns
df_asrs_raw = pd.read_csv("../data/ASRS_DBOnline.csv", header=[0, 1])

# Pull out the three columns we need using their MultiIndex keys
df_asrs = pd.DataFrame({
    "date":      df_asrs_raw[("Time",  "Date")],
    "locale":    df_asrs_raw[("Place", "Locale Reference")],
    "narrative": df_asrs_raw[("Report 1", "Narrative")],
})

# Date is YYYYMM (e.g. 202501) -- last 2 digits give the month
df_asrs["month"] = df_asrs["date"].astype(str).str[-2:].astype(int)

# Locale is "SFO.Airport", "ZZZ.Airport", etc.
# "ZZZ" means the airport is anonymised -- we cannot match those rows
df_asrs["airport_code"] = df_asrs["locale"].astype(str).str.split(".").str[0]
df_asrs = df_asrs[df_asrs["airport_code"] != "ZZZ"].copy()

print(f"Usable ASRS records (known airport): {len(df_asrs)}")
print("Airport codes found:", df_asrs["airport_code"].unique().tolist())

# Score each narrative with the zero-shot LLM
df_asrs["risk_score"] = df_asrs["narrative"].apply(score_narrative)

# Map IATA codes to the full airport names used in merged_flights.csv
CODE_TO_NAME = {
    "SFO": "San Francisco International",
    "LAX": "Los Angeles International",
    "ORD": "Chicago O Hare International",
    "ATL": "Atlanta Municipal",
    "DEN": "Denver International",
    "DFW": "Dallas/Fort Worth International",
    "JFK": "John F Kennedy International",
    "LAS": "McCarran International",
    "MCO": "Orlando International",
    "SEA": "Seattle-Tacoma International",
    "CTY": "Cross City",
    "X04": "Lake Wales Municipal",
}
df_asrs["DEPARTING_AIRPORT"] = df_asrs["airport_code"].map(CODE_TO_NAME)
df_asrs = df_asrs.dropna(subset=["DEPARTING_AIRPORT"])
df_asrs["MONTH"] = df_asrs["month"]

# Aggregate: mean risk score per (airport, month)
asrs_agg = (
    df_asrs
    .groupby(["DEPARTING_AIRPORT", "MONTH"])["risk_score"]
    .mean()
    .reset_index()
    .rename(columns={"risk_score": "asrs_risk_score"})
)
asrs_agg["asrs_risk_score"] = asrs_agg["asrs_risk_score"].round(4)

# Left join onto main dataset; unmatched flights get score 0
df = df.merge(asrs_agg, on=["DEPARTING_AIRPORT", "MONTH"], how="left")
df["asrs_risk_score"] = df["asrs_risk_score"].fillna(0.0)

print(f"asrs_risk_score column added.")
print(f"Non-zero entries: {(df['asrs_risk_score'] > 0).sum()} / {len(df)}")
print(df["asrs_risk_score"].describe().round(4))
"""

# Find the broken cell by looking for the old groupby pattern
for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "groupby" in src and "airport" in src and cell["cell_type"] == "code":
        cell["source"] = new_source
        print("Patched cell.")
        break

with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved.")
