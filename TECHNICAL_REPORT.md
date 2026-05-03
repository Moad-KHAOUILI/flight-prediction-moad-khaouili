# Technical Report — Flight Delay & Passenger Count Prediction

**Author:** Moad Khaouili  
**Repository:** https://github.com/Moad-KHAOUILI/flight-prediction-moad-khaouili  
**Date:** May 2026

---

## 1. Introduction

### Problem Statement

Flight delays are a persistent and costly problem for the global aviation industry. In 2023 alone, U.S. domestic delays cost airlines an estimated **$33 billion** in direct operating costs and passengers countless hours of lost time. Accurately predicting whether a flight will be delayed — and by how many minutes — allows airlines to proactively reroute crews, alert passengers, and optimise gate assignments. A complementary prediction of **passenger count** helps airlines plan ground services, catering, and fuel loads more efficiently.

### Motivation

Classic rule-based delay models rely on simple threshold heuristics (e.g., "if weather score > X, flag as delayed"). These approaches fail to capture the complex, non-linear interactions between carrier performance, air-traffic congestion, seasonal patterns, and cascading delay propagation across a flight network. Machine learning offers a principled way to learn these patterns from historical data at scale.

### Chosen Approach

This project applies a **hybrid ML pipeline**:

1. **XGBoost / LightGBM** — fast, interpretable gradient-boosted tree baselines for both classification (delayed / on-time) and regression (delay minutes, passenger count).
2. **PyTorch Transformer Encoder** — a sequence model that treats a route's recent flight history as a temporal sequence, using multi-head self-attention to capture long-range dependencies between consecutive flights on the same route.
3. **OpenAI GPT API** — an LLM layer that ingests SHAP feature-importance values and generates plain-English operational summaries for airline staff, fulfilling the *Large Language Model* requirement of the project scope.

The combination places this work beyond basic supervised learning by introducing attention-based deep learning and generative AI for insight communication.

---

## 2. Data

### Dataset Description

The primary dataset is the **Bureau of Transportation Statistics (BTS) On-Time Performance** table, covering U.S. domestic scheduled flights from 2019 to 2023 (~60 million records). A curated 5.8-million-row subset from Kaggle (dataset: `usdot/flight-delays`) was also used for rapid prototyping.

**Key columns:**

| Feature | Type | Description |
|---------|------|-------------|
| `carrier` | Categorical | Two-letter IATA airline code |
| `origin` / `dest` | Categorical | Origin and destination airport codes |
| `crs_dep_time` | Integer | Scheduled departure time (HHMM) |
| `distance` | Numeric | Route distance in miles |
| `day_of_week` / `month` | Integer | Temporal context |
| `dep_delay` | Numeric | **Target A** – actual departure delay in minutes |
| `arr_delay` | Numeric | **Target B** – arrival delay in minutes |
| `passengers` | Numeric | **Target C** – number of passengers boarded |
| `carrier_delay`, `weather_delay`, `nas_delay` | Numeric | Breakdown of delay cause (used as auxiliary features during training only) |

### Preprocessing Steps

1. **Missing-value handling:** Rows with null `arr_delay` (cancelled/diverted flights) were removed for the regression task and kept as a separate "cancelled" class for multi-class classification experiments.
2. **Outlier clipping:** Delay values beyond the 99th percentile (>330 min) were clipped to reduce the influence of rare extreme events.
3. **Cyclic encoding:** `month`, `day_of_week`, and `dep_hour` were encoded as sine/cosine pairs to preserve circular continuity.
4. **Label encoding + embeddings:** High-cardinality categoricals (`carrier`, `origin`, `dest`) were label-encoded for tree models and mapped to learnable embeddings for the Transformer model.
5. **Sequence construction:** For the Transformer, each sample consists of the last **14 consecutive flights** on the same route, sorted by scheduled departure time. Padding masks were applied for routes with fewer than 14 historical records.
6. **Train / validation / test split:** Data was split chronologically (2019–2021 train, 2022 validation, 2023 test) to prevent data leakage from temporal autocorrelation.

### Challenges

- **Class imbalance:** Only ~20% of flights are delayed by more than 15 minutes. Addressed with `scale_pos_weight` in XGBoost and a weighted cross-entropy loss in the Transformer.
- **Data volume:** The full BTS table required chunked loading with `pandas.read_csv(chunksize=...)` and Parquet serialisation for efficient re-reads.
- **Missing passenger counts:** The BTS on-time table does not directly include passenger counts; these were joined from the T-100 Domestic Segment table using `(carrier, origin, dest, year, month)` as the composite key, resulting in a ~15% match-failure rate that was imputed with carrier/route monthly medians.

---

## 3. Model & Methods

### 3.1 XGBoost / LightGBM Baseline

Two gradient-boosted tree models were trained:

- **XGBoostClassifier** for binary delay classification (threshold: ≥15 min late → delayed).
- **LightGBM Regressor** for delay-minutes regression and for passenger-count regression.

Hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`) were tuned with **Optuna** using 50 trials and 5-fold time-series cross-validation. SHAP was used post-hoc to rank feature importances and diagnose the model.

### 3.2 PyTorch Transformer Encoder

The Transformer model treats each route's flight history as a sequence:

```
Input: (batch_size, seq_len=14, n_features=32)
  → Linear projection → d_model = 128
  → Sinusoidal positional encoding
  → 4 × TransformerEncoderLayer (nhead=8, dim_feedforward=512, dropout=0.1)
  → [CLS]-token pooling
  → Task head A: Linear(128 → 1) + sigmoid  [delay classification]
  → Task head B: Linear(128 → 1)             [passenger count regression]
```

Training used AdamW (lr = 1e-4, weight_decay = 1e-2) with a cosine annealing schedule, mixed-precision (PyTorch AMP), and early stopping (patience = 5 epochs on validation loss).

**Why Transformers?** Self-attention allows the model to identify which past flights on a route (e.g., the first morning departure) are most predictive of current delay risk, modelling propagation effects that tree models cannot capture across a sequence.

### 3.3 OpenAI LLM Insight Generation

After training, SHAP values were computed for a stratified sample of 10,000 test records. A structured prompt was constructed:

```
System: You are an aviation operations analyst. Given SHAP feature importance
        data for a flight delay model, produce a JSON report with:
        (1) top 5 delay drivers, (2) carrier-specific recommendations,
        (3) seasonal risk periods.

User: [serialised SHAP summary as JSON]
```

GPT-4o responses were parsed and saved as `reports/llm_insights_report.json`. This component fulfils the **LLM** requirement by using the OpenAI API for a task that goes beyond prediction — generating actionable, human-readable operational guidance.

---

## 4. Results & Evaluation

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| XGBoost baseline | Delay classification | ROC-AUC | 0.87 |
| XGBoost baseline | Delay classification | F1 (delayed class) | 0.72 |
| LightGBM | Delay minutes (regression) | MAE | 11.3 min |
| LightGBM | Passenger count (regression) | RMSE | 18.4 pax |
| Transformer | Delay classification | ROC-AUC | **0.89** |
| Transformer | Passenger count (regression) | RMSE | **15.1 pax** |

**Key findings:**

- The Transformer outperforms LightGBM on both tasks, confirming that sequential context (prior flights on the same route) carries predictive signal beyond single-flight tabular features.
- **`carrier_delay`** and **`nas_delay`** are the top SHAP contributors to delay magnitude, followed by `dep_hour` (late-night flights accumulate less delay than early-morning push-backs that cascade throughout the day).
- Passenger count is most strongly driven by `distance` and `month`, with a sharp dip in February and surge in June/July consistent with U.S. travel seasonality.
- The LLM insight layer correctly identified Southwest Airlines and Newark Liberty (EWR) as high-risk carrier/airport combinations in the 2023 test period — consistent with reported operational issues.

---

## 5. Contributions

**Moad Khaouili** completed the full project independently.

| Component | Work done | Tools / GenAI used |
|-----------|-----------|---------------------|
| Data pipeline | Downloaded BTS & Kaggle data; wrote preprocessing & feature-engineering scripts | pandas, NumPy |
| Baseline models | Designed XGBoost/LightGBM training loop; ran Optuna hyperparameter search | scikit-learn, XGBoost, LightGBM, Optuna |
| Transformer model | Designed and implemented the sequence Transformer in PyTorch from scratch | PyTorch (new-to-me technology) |
| LLM integration | Wrote prompt engineering logic and OpenAI API integration | OpenAI Python SDK (new-to-me technology) |
| Evaluation & SHAP | Computed SHAP values; generated evaluation plots | SHAP, Matplotlib, Seaborn |
| Documentation | Wrote README and this report | — |
| GenAI assistance | Used GitHub Copilot for boilerplate code (dataset loading, training loops); used ChatGPT to help debug PyTorch tensor shape errors and draft prompt templates for the LLM insights module | GitHub Copilot, ChatGPT |

**Online resources used:** BTS dataset documentation, the "Attention Is All You Need" paper (Vaswani et al., 2017), PyTorch official Transformer tutorial, OpenAI API cookbook examples for structured output.

---

## 6. Challenges & Future Work

### Challenges Encountered

- **Passenger data join complexity:** Merging BTS on-time and T-100 segment tables required careful handling of time granularities (monthly vs. daily) and airline code changes over time.
- **Sequence alignment:** Constructing fixed-length flight sequences per route with correct chronological ordering was non-trivial; routes with sparse history needed special padding handling.
- **LLM prompt reliability:** Early prompts produced inconsistently structured JSON; few-shot examples and explicit schema constraints in the system prompt were necessary to achieve reliable parsing.
- **Compute cost:** Training the Transformer on the full 60 M-row dataset required ~6 hours on a single A100 GPU. Developing on the 5.8 M Kaggle subset first significantly shortened the iteration cycle.

### Future Work

1. **Real-time weather integration:** Pull METAR/TAF data from the Aviation Weather Center API as live features to substantially improve short-horizon predictions.
2. **REST API & dashboard:** Wrap `predict.py` in a FastAPI service and build a Streamlit front-end for airline operations staff.
3. **Fine-tuned open-source LLM:** Replace the OpenAI API call with a locally hosted, fine-tuned LLaMA 3 model trained on aviation-domain incident reports to remove the API cost and latency.
4. **Causal modelling:** Explore causal inference (DoWhy) to move from correlation-based insights to actionable root-cause analysis of delay propagation.
5. **International expansion:** Extend the dataset to Eurocontrol and ICAO data for European and intercontinental routes.
