# Retail Demand Forecasting

An end-to-end **machine learning and MLOps project for next-day retail demand forecasting**.

The project takes historical store-product sales data, engineers time-aware features, trains and evaluates multiple regression models, tracks experiments with MLflow, versions the dataset with DVC, and exposes the trained model through a FastAPI service that can be deployed using Docker.

---

## Project Overview

Retail businesses need to estimate future product demand to make better decisions around:

* Inventory planning
* Stock replenishment
* Store-level operations
* Product availability
* Demand planning

The goal of this project is to predict:

> **How much demand should we expect for a particular product at a particular store on the next day?**

The prediction unit is:

```text
Store × Product × Date
```

For each store-product pair on day `t`, the model predicts the demand on day `t + 1`.

This makes the problem a **supervised regression problem with a time-dependent prediction target**.

---

## Problem Definition

### Input

Historical information for a store-product combination, including:

* Historical demand
* Units sold
* Units ordered
* Inventory level
* Price
* Competitor pricing
* Discount
* Promotion
* Weather condition
* Seasonality
* Store
* Product
* Date

### Target

```text
Next-day Demand
```

Formally:

```text
Target(t) = Demand(t + 1)
```

The target is created using a group-wise time shift:

```python
df.groupby(["Store ID", "Product ID"])["Demand"].shift(-1)
```

This ensures that the model learns to forecast future demand rather than reproduce the current day's demand.

---

## Dataset

The dataset contains approximately **76,000 observations** covering multiple:

* Stores
* Products
* Categories
* Regions
* Dates

The raw dataset is maintained using **DVC** rather than committing the complete dataset directly to Git.

### Main columns

| Column               | Description          |
| -------------------- | -------------------- |
| `Date`               | Observation date     |
| `Store ID`           | Store identifier     |
| `Product ID`         | Product identifier   |
| `Category`           | Product category     |
| `Region`             | Store region         |
| `Inventory Level`    | Available inventory  |
| `Units Sold`         | Units sold           |
| `Units Ordered`      | Units ordered        |
| `Price`              | Product price        |
| `Discount`           | Applied discount     |
| `Weather Condition`  | Weather information  |
| `Promotion`          | Promotion indicator  |
| `Competitor Pricing` | Competitor price     |
| `Seasonality`        | Seasonal information |
| `Epidemic`           | Epidemic indicator   |
| `Demand`             | Observed demand      |

The dataset contains no missing values.

---

# Machine Learning Approach

## 1. Exploratory Data Analysis

The initial analysis examined:

* Missing values
* Duplicate store-product-date combinations
* Demand distribution
* Outliers
* Temporal demand patterns
* Store-level behavior
* Product-level behavior
* Category-level behavior
* Regional behavior
* Seasonality
* Weekday/weekend behavior

The analysis did not reveal strong irregularities requiring aggressive preprocessing.

Instead of applying transformations simply because they are available, preprocessing decisions were based on the observed data.

---

## 2. Feature Engineering

The model uses historical information available before the prediction date.

### Demand lags

```text
demand_lag_1
demand_lag_7
demand_lag_14
demand_lag_28
```

These capture recent demand as well as weekly and longer-term patterns.

### Sales lags

```text
units_sold_lag_1
units_sold_lag_7
units_sold_lag_14
```

### Inventory lags

```text
inventory_lag_1
inventory_lag_7
```

### Order lags

```text
orders_lag_1
orders_lag_7
```

### Rolling demand statistics

```text
demand_rolling_mean_7
demand_rolling_std_7
demand_rolling_mean_14
demand_rolling_mean_28
```

Rolling statistics are calculated using previous observations only to avoid future information leaking into the features.

For example:

```python
demand_history = history["Demand"].shift(1)
```

before calculating rolling statistics.

### Calendar features

```text
day_of_week
day_of_month
week_of_year
month
quarter
is_weekend
```

### Price features

```text
price_difference
price_ratio
```

These capture the relationship between the product's price and competitor pricing.

---

# Avoiding Data Leakage

Because this is a forecasting problem, random train-test splitting would allow future observations to influence the past.

Instead, the dataset is split chronologically.

```text
|---------------- Training ----------------|------ Validation ------|------ Test ------|
                                      Time →
```

The split is approximately:

```text
70% Training
15% Validation
15% Test
```

The split is performed according to chronological order rather than randomly.

The validation set is used to select the best model.

The test set is evaluated only after model selection.

---

# Models

Three regression models are evaluated:

### Linear Regression

Used as a simple baseline.

### Random Forest

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

### XGBoost

```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

Each model is evaluated using the same preprocessing and chronological validation set.

The model with the lowest validation **MAE** is selected.

---

# Evaluation Metrics

The primary metric is:

### MAE — Mean Absolute Error

```text
MAE = mean(|actual - predicted|)
```

MAE is used as the primary metric because it is easy to interpret in terms of demand units.

Two additional metrics are tracked:

### RMSE

Penalizes larger prediction errors more strongly.

### sMAPE

Provides a scale-independent percentage-based measure of forecast error.

The project therefore evaluates:

```text
Primary:
    MAE

Secondary:
    RMSE
    sMAPE
```

---

# Experiment Tracking with MLflow

**MLflow** is used to track model experiments.

Each model run records:

* Model name
* Model parameters
* Validation MAE
* Validation RMSE
* Validation sMAPE
* Test metrics for the selected model
* Serialized model artifact

The experiment is organized under:

```text
retail-demand-forecasting
```

This makes it possible to compare different models without relying on manually recorded results.

Run MLflow locally with:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Data Versioning with DVC

The raw dataset is versioned using **DVC**.

The dataset itself is not treated as a normal Git-tracked file.

Instead:

```text
Git
 │
 ├── Code
 ├── Configuration
 └── sales_data.csv.dvc
             │
             ▼
            DVC
             │
             ▼
       Actual dataset
```

The raw dataset can be tracked with:

```bash
dvc add data/raw/sales_data.csv
```

The complete pipeline can be reproduced with:

```bash
dvc repro
```

The DVC pipeline contains two main stages:

```text
Raw Data
   │
   ▼
Feature Engineering
   │
   ▼
Processed Dataset
   │
   ▼
Model Training
   │
   ▼
Trained Model
```

---

# FastAPI Prediction Service

The trained model is exposed through a **FastAPI** application.

The API intentionally accepts business-level inputs instead of requiring users to manually calculate model features.

### Request

```http
POST /predict
```

```json
{
  "store_id": "S001",
  "product_id": "P0001"
}
```

The API:

1. Finds historical data for the store-product pair.
2. Sorts the historical observations by date.
3. Builds the same features used during training.
4. Creates the next prediction date.
5. Loads the trained model.
6. Generates the next-day demand prediction.
7. Returns the prediction.

### Response

```json
{
  "store_id": "S001",
  "product_id": "P0001",
  "prediction_date": "2026-10-06",
  "predicted_demand": 104.82735443115234
}
```

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### API Documentation

When the API is running:

```text
http://localhost:8000/docs
```

FastAPI automatically provides an interactive Swagger UI.

---

# Docker

The API is containerized using Docker.

The container packages:

```text
FastAPI application
        +
Inference code
        +
Trained model
        +
Required raw historical data
        +
Python dependencies
```

Build the image:

```bash
docker build -t retail-demand-forecasting .
```

Run the container:

```bash
docker run --name retail-demand-api -p 8000:8000 retail-demand-forecasting
```

The API is then available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Check container logs:

```bash
docker logs retail-demand-api
```

---

# Project Structure

```text
retail-demand-forecasting/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   ├── demand_forecasting_model.joblib
│   └── test_predictions.csv
│
├── data/
│   ├── raw/
│   │   ├── sales_data.csv
│   │   ├── sales_data.csv.dvc
│   │   └── .gitignore
│   │
│   └── processed/
│       └── model_data.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── inference.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── mlflow_utils.py
│   │   └── train.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py
│
├── tests/
│   └── test_features.py
│
├── artifacts/
│
├── dvc.yaml
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

---

# Running the Project Locally

## 1. Clone the repository

```bash
git clone <repository-url>
cd retail-demand-forecasting
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For reproducing the exact development environment:

```bash
pip install -r requirements-lock.txt
```

## 4. Reproduce the ML pipeline

```bash
dvc repro
```

This runs:

```text
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Artifact
```

## 5. Start the API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

# Running with Docker

Build:

```bash
docker build -t retail-demand-forecasting .
```

Run:

```bash
docker run --name retail-demand-api -p 8000:8000 retail-demand-forecasting
```

Then visit:

```text
http://localhost:8000/docs
```

---

# Testing

Run the test suite using:

```bash
pytest
```

Feature engineering tests are located under:

```text
tests/
```

The purpose of the tests is to verify that the feature generation logic remains consistent and does not silently break the model input contract.

---

# Reproducibility

The project uses multiple layers of versioning and reproducibility:

| Component             | Tool                    |
| --------------------- | ----------------------- |
| Source code           | Git                     |
| Dataset               | DVC                     |
| Experiments           | MLflow                  |
| Python dependencies   | `requirements.txt`      |
| Exact environment     | `requirements-lock.txt` |
| Model artifact        | Joblib                  |
| Application packaging | Docker                  |

This creates a reproducible path from:

```text
Data
 ↓
Features
 ↓
Experiments
 ↓
Model
 ↓
API
 ↓
Container
```

---

# MLOps Workflow

The overall workflow is:

```text
                    ┌──────────────┐
                    │  Raw Dataset │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     DVC      │
                    │ Data Version │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     EDA      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Feature    │
                    │ Engineering  │
                    └──────┬───────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Model Training     │
                 │ Linear Regression  │
                 │ Random Forest      │
                 │ XGBoost            │
                 └─────────┬──────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    MLflow    │
                    │ Experiment   │
                    │   Tracking   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Best Model   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   FastAPI    │
                    │ Prediction   │
                    │    Service   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Docker    │
                    │  Container   │
                    └──────────────┘
```

---

# Key Engineering Decisions

### Why chronological splitting?

Demand forecasting is inherently time-dependent. A random split could expose the model to future patterns while training on earlier observations.

### Why lag and rolling features?

Past demand is one of the strongest sources of information for predicting future demand. Lag and rolling features allow the model to capture recent demand behavior and recurring patterns.

### Why MAE as the primary metric?

MAE directly represents the average absolute error in demand units and is easier to interpret for operational forecasting.

### Why MLflow?

Comparing multiple models manually becomes difficult as experiments increase. MLflow provides a structured record of parameters, metrics, and model artifacts.

### Why DVC?

Large datasets should not be managed directly through Git. DVC keeps data versioning connected to the Git history while keeping the repository lightweight.

### Why FastAPI?

The trained model should not remain accessible only through a notebook or Python script. FastAPI turns the model into a usable prediction service.

### Why Docker?

Docker packages the application and its runtime dependencies together, reducing environment-related differences between development and deployment.

---

# Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### MLOps

* MLflow
* DVC
* Git / GitHub

### API

* FastAPI
* Pydantic
* Uvicorn

### Deployment

* Docker

### Visualization / Analysis

* Matplotlib
* Jupyter Notebook

---

# Current Limitations

This is a compact end-to-end MLOps project rather than a production retail forecasting platform.

Current limitations include:

* The model is trained on a static historical dataset.
* The API reads historical data from the packaged dataset.
* There is no automated cloud deployment.
* There is no live data ingestion pipeline.
* Model monitoring is not yet implemented.
* Retraining is currently triggered manually through the pipeline.
* The prediction service is not horizontally deployed.
* No authentication or authorization layer is implemented.

These limitations are intentional so that the project remains understandable while demonstrating the complete ML lifecycle.

---

# Future Improvements

Potential next steps include:

* MLflow Model Registry
* DagsHub integration
* Automated CI/CD
* Scheduled model retraining
* Data drift detection
* Prediction monitoring
* Model performance monitoring
* Cloud deployment
* Automated feature pipelines
* Live inventory/sales integration
* API authentication
* Batch forecasting
* Multi-day forecasting
* Automated model promotion

---

# What This Project Demonstrates

This project goes beyond training a regression model.

It demonstrates the complete path from a business problem to a usable ML system:

```text
Business Problem
      ↓
Data Understanding
      ↓
Time-Aware Feature Engineering
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Experiment Tracking
      ↓
Data Versioning
      ↓
Model Packaging
      ↓
API Serving
      ↓
Containerization
```

The main objective was to understand **how an ML model moves from experimentation into a reproducible and deployable system**, rather than focusing only on model accuracy.

---

## Author

**Rohit Gupta**

AI/ML Engineer focused on:

```text
Machine Learning
Generative AI
LLMs & RAG
AI Agents
Data & AI Engineering
MLOps
```

GitHub: `https://github.com/Rohit991371`
