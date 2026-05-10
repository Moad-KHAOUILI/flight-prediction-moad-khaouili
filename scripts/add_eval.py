import json, uuid

def code_cell(src):
    return {"cell_type":"code","execution_count":None,
            "id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

MD_EVAL = """## 5. Extended Evaluation

Additional diagnostics for both models:
- **ROC curves** -- trade-off between true positive rate and false positive rate at all thresholds
- **Precision-Recall curves** -- especially informative for the imbalanced delay dataset (~20% positive)
- **Threshold sweep** -- find the decision threshold that maximises F1 on the test set
- **Permutation feature importance** -- which features matter most for each model"""

# ── ROC + PR curves for both models ─────────────────────────────────────────
CURVES = """from sklearn.metrics import roc_curve, precision_recall_curve, auc

def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()

probs_delay = get_probs(model_delay, X_te)
probs_ob    = get_probs(model_ob,    X_te_ob)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── ROC ──
for (probs, y_true, label, color) in [
    (probs_delay, y_te,    "Delay (AUC={:.3f})".format(roc_auc_score(y_te,    probs_delay)), "steelblue"),
    (probs_ob,    y_te_ob, "Overbooking (AUC={:.3f})".format(roc_auc_score(y_te_ob, probs_ob)), "darkorange"),
]:
    fpr, tpr, _ = roc_curve(y_true, probs)
    axes[0].plot(fpr, tpr, label=label, color=color, lw=2)
axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

# ── Precision-Recall ──
for (probs, y_true, label, color) in [
    (probs_delay, y_te,    "Delay", "steelblue"),
    (probs_ob,    y_te_ob, "Overbooking", "darkorange"),
]:
    prec, rec, _ = precision_recall_curve(y_true, probs)
    axes[1].plot(rec, prec, label=f"{label} (AP={auc(rec,prec):.3f})", color=color, lw=2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()"""

# ── Threshold sweep ──────────────────────────────────────────────────────────
THRESH = """from sklearn.metrics import f1_score

def threshold_sweep(probs, y_true, title):
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true.astype(int), (probs >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, f1s, lw=2, color="steelblue")
    plt.axvline(best_t, color="red", linestyle="--",
                label=f"Best threshold={best_t:.2f}  F1={best_f1:.4f}")
    plt.xlabel("Decision Threshold"); plt.ylabel("F1 Score (positive class)")
    plt.title(f"Threshold vs F1 -- {title}"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"{title}: best threshold={best_t:.2f}, F1={best_f1:.4f}")
    preds_opt = (probs >= best_t).astype(int)
    print(classification_report(y_true.astype(int), preds_opt,
                                target_names=["No Delay", "Delayed"], digits=4))

threshold_sweep(probs_delay, y_te,    "Flight Delay MLP")
threshold_sweep(probs_ob,    y_te_ob, "Overbooking MLP")"""

# ── Permutation importance ───────────────────────────────────────────────────
PERM = """from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

class TorchWrapper(BaseEstimator, ClassifierMixin):
    \"\"\"Thin sklearn-compatible wrapper around a PyTorch binary classifier.\"\"\"
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        return self
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.tensor(X, dtype=torch.float32))).numpy()
        return (p >= 0.5).astype(int)
    def score(self, X, y):
        return roc_auc_score(y, self._proba(X))
    def _proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(X, dtype=torch.float32))).numpy()

def plot_importance(model, X_te, y_te, feature_names, title, top_n=15):
    wrapper = TorchWrapper(model)
    result  = permutation_importance(wrapper, X_te, y_te.astype(int),
                                     n_repeats=10, random_state=42,
                                     scoring="roc_auc")
    idx = np.argsort(result.importances_mean)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], result.importances_mean[idx],
            xerr=result.importances_std[idx], color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean ROC-AUC drop"); ax.set_title(f"Permutation Importance -- {title}")
    ax.grid(axis="x", alpha=0.3); plt.tight_layout(); plt.show()

print("Computing permutation importance for Delay model (takes ~1 min)...")
plot_importance(model_delay, X_te,    y_te,    feature_cols_delay, "Flight Delay MLP")

print("Computing permutation importance for Overbooking model...")
plot_importance(model_ob,    X_te_ob, y_te_ob, feature_cols_ob,    "Overbooking MLP")"""

new_cells = [
    md_cell(MD_EVAL),
    code_cell(CURVES),
    code_cell(THRESH),
    code_cell(PERM),
]

nb["cells"].extend(new_cells)
with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Added {len(new_cells)} cells. Total: {len(nb['cells'])}")
