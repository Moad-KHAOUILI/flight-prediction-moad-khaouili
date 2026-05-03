# ✈️ Flight Delay & Passenger Count Prediction

An AI-driven solution that predicts **flight delays** and **passenger counts** using a hybrid machine-learning pipeline — combining gradient-boosted trees with a PyTorch Transformer (attention-based) model and an OpenAI LLM layer for automated insight generation.

---

## 📌 Project Overview

Flight disruptions cost airlines and passengers billions of dollars every year. This project builds a predictive system that:

1. **Predicts whether a flight will be delayed** (binary classification) and **by how many minutes** (regression).
2. **Forecasts the number of passengers** on a given flight (regression).
3. **Surfaces human-readable insights** about delay drivers using the OpenAI API.

The model pipeline goes beyond basic supervised learning by incorporating:
- A **Transformer encoder with multi-head self-attention** (PyTorch) for temporal flight-sequence modelling.
- **OpenAI GPT API** calls that translate model outputs and SHAP feature-importance values into plain-English summaries for airline operations staff.

---

## 📂 Repository Structure

```
flight-prediction-moad-khaouili/
├── data/
│   ├── raw/                  # Original BTS / Kaggle CSV files (not committed)
│   └── processed/            # Cleaned & feature-engineered files (not committed)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_xgboost_baseline.ipynb
│   ├── 04_transformer_model.ipynb
│   └── 05_llm_insights.ipynb
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   ├── train_xgb.py          # XGBoost/LightGBM training script
│   ├── train_transformer.py  # PyTorch Transformer training script
│   ├── predict.py            # Inference script
│   └── llm_insights.py       # OpenAI API integration for insight generation
├── models/
│   └── (saved model artefacts – not committed, see setup below)
├── reports/
│   └── figures/              # Plots generated during evaluation
├── requirements.txt
├── TECHNICAL_REPORT.md
└── README.md
```

---

## 🗄️ Dataset

| Source | Description |
|--------|-------------|
| [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ) | U.S. domestic on-time performance data (2019–2023, ~60 M rows) |
| [Kaggle – Flight Delay Dataset](https://www.kaggle.com/datasets/usdot/flight-delays) | Curated 5.8 M-row subset with carrier, origin, destination, scheduled times, delay minutes, and passenger counts |

**Key features used:**

`carrier`, `origin`, `destination`, `scheduled_departure`, `scheduled_arrival`, `distance`, `day_of_week`, `month`, `is_holiday`, `weather_delay`, `carrier_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay`, `passenger_count`

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- `pip` or `conda`
- An [OpenAI API key](https://platform.openai.com/account/api-keys) (only required for the LLM-insights module)
- (Optional) CUDA-capable GPU for faster Transformer training

### 1. Clone the repository

```bash
git clone https://github.com/Moad-KHAOUILI/flight-prediction-moad-khaouili.git
cd flight-prediction-moad-khaouili
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

Download the Kaggle dataset (requires a Kaggle account):

```bash
pip install kaggle
kaggle datasets download -d usdot/flight-delays -p data/raw --unzip
```

Or manually download the BTS CSV files and place them in `data/raw/`.

### 5. Set environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...          # Required for llm_insights.py
```

### 6. Run the pipeline

```bash
# Step 1 – Preprocess raw data
python src/preprocess.py

# Step 2 – Train XGBoost baseline
python src/train_xgb.py

# Step 3 – Train Transformer model
python src/train_transformer.py --epochs 20 --batch_size 512

# Step 4 – Generate LLM-powered insights report
python src/llm_insights.py

# Step 5 – Run inference on new data
python src/predict.py --input data/processed/test.csv --output predictions.csv
```

---

## 🧠 Model Architecture

### 1. XGBoost / LightGBM Baseline
- **Task A:** Binary classification (delayed / on-time) using XGBoostClassifier.
- **Task B:** Regression (delay minutes & passenger count) using LightGBM.
- Hyperparameters tuned with Optuna (50 trials, 5-fold CV).

### 2. PyTorch Transformer Encoder
- Sequence of past flight events per route is encoded with positional embeddings.
- 4-layer Transformer encoder with 8-head multi-head self-attention (d_model = 128).
- Task-specific linear heads for delay classification and passenger-count regression.
- Trained with mixed-precision (AMP) on CUDA; early stopping on validation loss.

### 3. OpenAI LLM Insights
- SHAP values from the XGBoost model are serialised and injected into a GPT-4o prompt.
- The LLM produces a structured JSON report of key delay drivers and actionable recommendations for each airport/carrier pair.

---

## 📊 Results

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| XGBoost (baseline) | Delay classification | ROC-AUC | 0.87 |
| LightGBM | Delay minutes regression | MAE (min) | 11.3 |
| LightGBM | Passenger count regression | RMSE | 18.4 |
| Transformer | Delay classification | ROC-AUC | 0.89 |
| Transformer | Passenger count regression | RMSE | 15.1 |

> Full evaluation plots (confusion matrices, SHAP beeswarm, learning curves) are in `reports/figures/`.

---

## ⚙️ Requirements

```
# requirements.txt (representative – see file for pinned versions)
pandas
numpy
scikit-learn
xgboost
lightgbm
optuna
torch
shap
openai
python-dotenv
matplotlib
seaborn
jupyter
```

---

## 📄 Technical Report

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the full ~2-page technical report covering the problem statement, data, methods, results, contributions, and future work.

---

## 🤝 Contributions

| Contributor | Work |
|-------------|------|
| **Moad Khaouili** | Full project — data pipeline, model design & training, LLM integration, evaluation, documentation |

---

## 🔮 Future Work

- Incorporate real-time weather API data as live features.
- Deploy the model as a REST API (FastAPI) with a Streamlit dashboard.
- Fine-tune a smaller open-source LLM (LLaMA 3) on airline-domain text for offline insight generation.
- Expand to international routes using OpenSky Network data.

---

## 📜 License

This project is for academic use. Data sourced from publicly available BTS and Kaggle datasets.