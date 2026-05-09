# Technical report for flight predictor and passengers count

## 1. Introduction

**Firstly, this is not the finished technical report, I will keep pushing after I make any changes to the repository (link below), you're free to look at it.

### Problem Statement

Flight delays and overbooking are two of the most common issues in the airline industry, affecting millions of passengers every year. The goal of this project is to build two prediction models: one that predicts whether a flight will be delayed by more than 15 minutes, and one that predicts whether a flight will be overbooked. Both models are trained on a combination of flight data, weather conditions, and booking statistics.

### Motivation

Delays and overbooking are usually treated as separate problems, but they share a lot of the same underlying factors airline, route, time of year, airport congestion, and weather. By working on both together, we can build a more complete picture of what makes a flight go wrong.

### Chosen Approach

Two separate binary classification models, trained on the same merged dataset:

**Model 1 — Flight Delay Prediction**
- Target: will a flight arrive more than 15 minutes late? (yes/no)
- Features: airline, route, time of day, day of week, month, weather conditions, airport congestion, propagated delay from previous flight, holiday flag, LLM-based incident risk score

**Model 2 — Overbooking Prediction**
- Target: is a flight overbooked? (yes/no)
- Features: airline, month, load factor, seats, passengers, price category, route, geo region

**Modelling strategy for both:**
- Both models will use the same training pipeline, but using different feature sets. The data is split into training, validation, and test sets. Class weights are applied during training to handle potential class imbalance, and early stopping is used to prevent overfitting. Performance is evaluated on the test set using a confusion matrix.

### Link to repo

- repo: https://github.com/Moad-KHAOUILI/flight-prediction-moad-khaouili

### 2. Data

- Dataset 1: 2019 Airline Delays with Weather
- Dataset 2: Flight Bookings
- Dataset 3: NASA ASRS Aviation Safety Reports
- Merged Dataset: Merged on airline + month


### Dataset Description

- Dataset 1: 2019 Airline delays with weather contains around 6 million flight records from 2019. Includes flight information such as airline, origin, destination, and scheduled times, along with weather conditions at both airports (temperature, wind, precipitation, snow), airport congestion metrics, and detailed delay reasons. This dataset provides the target variable for the delay prediction model.

- Dataset 2: Flight bookings contains around 366,000 records with booking-level information per airline per month. Includes number of seats, passengers boarded, load factor, price category, and a binary overbooking indicator. This dataset provides the target variable for the overbooking prediction model.

- Dataset 3: NASA ASRS Aviation Safety Reports contains free-text incident reports submitted voluntarily by pilots, air traffic controllers, and ground staff. Each report describes a safety-related event or observation at a specific airport on a specific date. This dataset is used exclusively in the delay prediction model, where each report is scored by a large language model and the resulting risk score is added as a feature.

### preprocessing steps

- Merged Dataset: The two datasets are joined on airline and month, combining flight-level operational data with booking statistics into a single dataframe. Two additional features are computed after merging: a holiday flag indicating whether the flight falls on a US public holiday, and the delay of the previous flight operated by the same aircraft, used as a proxy for propagated delays.

- For Dataset 3, each incident report narrative is passed through a large language model which returns a risk score between 0 and 1 reflecting how likely the described event is to cause a flight delay. This score is then merged into the main dataset by matching on airport and date. Flights with no corresponding report on that day are assigned a score of 0.

### Challenges

most flights datasets are private and unpublished, so this is the best data i could come up with.

### 3. Model & Methods

- Both prediction tasks are treated as binary classification problems and solved using a Neural Network (Multi-Layer Perceptron). This type of network works well with structured tabular data and can pick up on complex patterns between features that simpler models would miss.

- Each model has three hidden layers with 128, 64, and 32 neurons, all using ReLU as the activation function. The output layer uses a Sigmoid function to produce a value between 0 and 1, which is then rounded to give the final yes/no prediction. Dropout layers are added to prevent the model from overfitting to the training data.

- Before training, categorical columns are converted to numbers using one-hot encoding, and numerical columns are rescaled so they are all on the same scale. Training is stopped early if the model stops improving on the validation set, and class weights are adjusted to make sure the model does not ignore the minority class in case the data is imbalanced.

- For the delay prediction model, an additional feature is derived from free-text airport incident reports using a large language model. Each report is passed to the LLM with a prompt asking it to rate the likelihood of the described event causing a flight delay, returning a score between 0 and 1. This score is appended as a numerical feature alongside the rest of the input. When no report is available for a given airport and date, the score defaults to 0.

### 4. Results & Evaluation

