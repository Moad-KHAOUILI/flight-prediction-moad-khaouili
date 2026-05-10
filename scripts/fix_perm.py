import json

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

new_src = """from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

class TorchWrapper(BaseEstimator, ClassifierMixin):
    \"\"\"Thin sklearn-compatible wrapper around a PyTorch binary classifier.\"\"\"
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])
    def fit(self, X, y):
        return self
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.tensor(X, dtype=torch.float32))).numpy()
        return np.column_stack([1 - p, p])

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

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "TorchWrapper" in src and "predict_proba" not in src:
        cell["source"] = new_src
        print("Patched permutation importance cell.")
        break

with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved.")
