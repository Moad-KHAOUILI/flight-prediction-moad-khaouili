import json, uuid

def code_cell(src):
    return {"cell_type":"code","execution_count":None,
            "id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

# ── Section 3 header ─────────────────────────────────────────────────────────
SEC3 = """## 3. Flight Delay Prediction

A **Multilayer Perceptron (MLP)** trained on `merged_flights.csv` predicts `DEP_DEL15` (1 = delayed >= 15 min, 0 = on time).

**Architecture:** Linear(128) \u2192 ReLU \u2192 Dropout(0.3) \u2192 Linear(64) \u2192 ReLU \u2192 Dropout(0.3) \u2192 Linear(32) \u2192 ReLU \u2192 Linear(1)  
**Loss:** BCEWithLogitsLoss with positive-class weight to handle class imbalance  
**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4), early stopping (patience=8)"""

# ── Feature engineering cell ─────────────────────────────────────────────────
DELAY_PREP = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay,
)

NUM_COLS = [
    "MONTH", "DAY_OF_WEEK", "DISTANCE_GROUP", "SEGMENT_NUMBER",
    "CONCURRENT_FLIGHTS", "NUMBER_OF_SEATS",
    "AIRPORT_FLIGHTS_MONTH", "AIRLINE_FLIGHTS_MONTH", "AIRLINE_AIRPORT_FLIGHTS_MONTH",
    "AVG_MONTHLY_PASS_AIRPORT", "AVG_MONTHLY_PASS_AIRLINE",
    "FLT_ATTENDANTS_PER_PASS", "GROUND_SERV_PER_PASS",
    "PLANE_AGE", "LATITUDE", "LONGITUDE",
    "PRCP", "SNOW", "SNWD", "TMAX", "AWND",
    "avg_load_factor", "avg_passengers", "overbooked_rate",
    "avg_seats_booked", "asrs_risk_score",
]
CAT_COLS = ["CARRIER_NAME", "DEP_TIME_BLK", "DEPARTING_AIRPORT"]
TARGET   = "DEP_DEL15"

df_delay = df.copy()

# Impute missing booking columns (18.6% of rows) with column mean
for col in ["avg_load_factor", "avg_passengers", "overbooked_rate", "avg_seats_booked"]:
    df_delay[col] = df_delay[col].fillna(df_delay[col].mean())

# If ASRS cell was skipped, add a zero column so the pipeline still works
if "asrs_risk_score" not in df_delay.columns:
    df_delay["asrs_risk_score"] = 0.0

# One-hot encode categoricals
df_delay = pd.get_dummies(df_delay, columns=CAT_COLS, drop_first=True, dtype=float)

ohe_cols = [c for c in df_delay.columns
            if any(c.startswith(p + "_") for p in CAT_COLS)]
feature_cols_delay = [c for c in NUM_COLS if c in df_delay.columns] + ohe_cols

X = df_delay[feature_cols_delay].astype(float).values
y = df_delay[TARGET].values.astype(float)

scaler_delay = StandardScaler()
X = scaler_delay.fit_transform(X)

# 70 / 15 / 15 train / val / test split (stratified)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
print(f"Delay rate  ->  train: {y_tr.mean():.3f}  |  test: {y_te.mean():.3f}")
print(f"Input dimension: {X_tr.shape[1]}")"""

# ── MLP + training + evaluation cell ────────────────────────────────────────
DELAY_TRAIN = """class MLP(nn.Module):
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


def train_mlp(X_tr, y_tr, X_val, y_val,
              epochs=80, patience=8, lr=1e-3, batch_size=256):
    model = MLP(X_tr.shape[1])
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                         dtype=torch.float32)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.tensor(X_tr,  dtype=torch.float32)
    yt = torch.tensor(y_tr,  dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    best_loss, no_improve, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()

        if val_loss < best_loss - 1e-6:
            best_loss, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop @ epoch {epoch}  (best val_loss={best_loss:.4f})")
                break
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, X_te, y_te, title=""):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32))
        probs  = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    print(f"\\n=== {title} ===")
    print(classification_report(y_te.astype(int), preds,
                                target_names=["No Delay", "Delayed"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_te, probs):.4f}")

    cm = confusion_matrix(y_te.astype(int), preds)
    ConfusionMatrixDisplay(cm, display_labels=["No Delay", "Delayed"]).plot(colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return probs


print("Training Flight Delay MLP...")
model_delay = train_mlp(X_tr, y_tr, X_val, y_val)
_ = evaluate_model(model_delay, X_te, y_te, title="Flight Delay MLP")"""

# ── Section 4 header ─────────────────────────────────────────────────────────
SEC4 = """## 4. Overbooking Prediction

A second MLP is trained on `flight_bookings.csv` to predict `overbooked` (1 = flight overbooked).  
The same architecture and training loop from Section 3 are reused. Features come from SFO monthly booking statistics."""

# ── Overbooking feature engineering cell ────────────────────────────────────
OB_PREP = """df_book = pd.read_csv("../data/flight_bookings.csv")
print(f"Bookings shape: {df_book.shape}")
print(df_book["overbooked"].value_counts(normalize=True).round(3))

NUM_OB    = ["month", "seats", "passengers", "load_factor"]
CAT_OB    = ["airline", "geo_summary", "geo_region", "terminal", "price_category"]
TARGET_OB = "overbooked"

df_b = df_book.copy()
df_b = pd.get_dummies(df_b, columns=CAT_OB, drop_first=True, dtype=float)

ohe_ob = [c for c in df_b.columns
          if any(c.startswith(p + "_") for p in CAT_OB)]
feature_cols_ob = [c for c in NUM_OB if c in df_b.columns] + ohe_ob

X_ob = df_b[feature_cols_ob].astype(float).values
y_ob = df_b[TARGET_OB].values.astype(float)

scaler_ob = StandardScaler()
X_ob = scaler_ob.fit_transform(X_ob)

X_tr_ob, X_tmp_ob, y_tr_ob, y_tmp_ob = train_test_split(
    X_ob, y_ob, test_size=0.30, random_state=42, stratify=y_ob)
X_val_ob, X_te_ob, y_val_ob, y_te_ob = train_test_split(
    X_tmp_ob, y_tmp_ob, test_size=0.50, random_state=42, stratify=y_tmp_ob)

print(f"Train: {X_tr_ob.shape}  Val: {X_val_ob.shape}  Test: {X_te_ob.shape}")
print(f"Overbooked rate  ->  train: {y_tr_ob.mean():.3f}  |  test: {y_te_ob.mean():.3f}")
print(f"Input dimension: {X_tr_ob.shape[1]}")"""

# ── Overbooking train + evaluate cell ────────────────────────────────────────
OB_TRAIN = """print("Training Overbooking MLP...")
model_ob = train_mlp(X_tr_ob, y_tr_ob, X_val_ob, y_val_ob, batch_size=1024)
_ = evaluate_model(model_ob, X_te_ob, y_te_ob, title="Overbooking MLP")"""

new_cells = [
    md_cell(SEC3),
    code_cell(DELAY_PREP),
    code_cell(DELAY_TRAIN),
    md_cell(SEC4),
    code_cell(OB_PREP),
    code_cell(OB_TRAIN),
]

nb["cells"].extend(new_cells)

with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(new_cells)} cells. Total: {len(nb['cells'])}")
