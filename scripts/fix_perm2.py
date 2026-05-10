import json

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

new_src = """def permutation_importance_manual(model, X_te, y_te, feature_names, title, top_n=15, n_repeats=8):
    \"\"\"Manual permutation importance: drop in ROC-AUC when each feature is shuffled.\"\"\"
    model.eval()
    def get_auc(X):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
        return roc_auc_score(y_te, p)

    baseline = get_auc(X_te)
    means, stds = [], []
    rng = np.random.default_rng(42)

    for j in range(X_te.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_te.copy()
            rng.shuffle(X_perm[:, j])
            drops.append(baseline - get_auc(X_perm))
        means.append(np.mean(drops))
        stds.append(np.std(drops))

    means, stds = np.array(means), np.array(stds)
    idx = np.argsort(means)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], means[idx],
            xerr=stds[idx], color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean ROC-AUC drop (baseline={:.4f})".format(baseline))
    ax.set_title(f"Permutation Importance -- {title}")
    ax.grid(axis="x", alpha=0.3); plt.tight_layout(); plt.show()

print("Computing permutation importance for Delay model...")
permutation_importance_manual(model_delay, X_te,    y_te,    feature_cols_delay, "Flight Delay MLP")

print("Computing permutation importance for Overbooking model...")
permutation_importance_manual(model_ob,    X_te_ob, y_te_ob, feature_cols_ob,    "Overbooking MLP")"""

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "TorchWrapper" in src or "permutation_importance" in src:
        cell["source"] = new_src
        print(f"Patched cell.")
        break

with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved.")
