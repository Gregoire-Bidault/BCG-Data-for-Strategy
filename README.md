# ClientCo ESG — Climate-Adjusted Yield Resilience Model

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Validation Metrics](#validation-metrics)
- [Stakeholders](#stakeholders)
- [Authors](#authors)

---

## Introduction

A publicly traded Food & Beverage producer (**ClientCo**) is seeking to turn climate risk into a **predictable operational variable**.

This project builds a **Climate-Adjusted Yield Resilience Model** that uses historical agricultural data and climate variables to forecast crop productivity for **barley** across 89 French departments.

By integrating climate signals (temperature anomalies, precipitation patterns, drought indices) with historical yield data, the model enables proactive supply-chain planning and resource allocation.


---

## Objectives

1. **Forecast crop productivity** — Predict barley yields at the department level for upcoming seasons.
2. **Reduce resource waste** — Leverage accurate forecasts to cut procurement and logistics waste.
3. **Quantify climate exposure** — Translate raw climate data into actionable drought and heat-stress metrics for each department.

---

## Quick Start

### Prerequisites

- Python 3.10+
- `pip` (or any preferred package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gregoire-Bidault/BCG-Data-for-Strategy.git

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
.venv/bin/streamlit run app.py

# 5. Launch Jupyter (optional – for EDA notebooks)
jupyter notebook
```

---

## Project Structure (WIP)

```
BCG-Data-for-Strategy/
│
├── data/
│   ├── raw/                 # Original, untouched datasets
│   │   ├── climate/         # Raw climate data (temperature, precipitation, …)
│   │   └── yields/          # Raw barley yield data
│   └── processed/           # Silver & Gold datasets
│
├── notebooks/               # Jupyter notebooks for EDA & prototyping
│
├── src/                     # Modular Python source code
│   ├── __init__.py
│   ├── bronze_to_silver.py
│   ├── silver_to_gold.py
│   ├── model_training.py        # Model fitting & hyper-parameter tuning
│   ├── validation.py            # Evaluation helpers (MAE, RMSE, …)
│   └── streamlit/               # Streamlit dashboard
│       ├── app.py               # Main entrypoint
│       └── pages/               # Multi-page app views
│
├── models/                  # Serialised trained models (.pkl / .joblib)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer               | Tool / Library                      | Purpose                                                       |
|---------------------|-------------------------------------|---------------------------------------------------------------|
| Data processing     | **Pandas**           | Fast ingestion & feature engineering |
| Modeling            | **XGBoost**, **Scikit-learn**                                 |                                                                |
| Visualisation       | **Matplotlib**, **Seaborn**         | Static & interactive charts for EDA and reporting              |
| Dashboard           | **Streamlit**                       | Interactive web app for exploring data & model results         |
| Notebooks           | **Jupyter** (ipykernel)             | Exploratory data analysis & rapid prototyping                  |

---

## Validation Metrics

Model performance is evaluated with yield-forecasting-specific metrics:

| Metric | Description |
|--------|-------------|
| **MAE**  | Mean Absolute Error — average magnitude of prediction errors |
| **RMSE** | Root Mean Squared Error — penalises large deviations more heavily |

---

## Stakeholders

| Role              | Responsibility          |
|-------------------|-------------------------|
| **CDO**           | Technical lead           |
| **Head of CSR**   | Strategic lead           |
| **Head of Finance** | Business logic         |

---

## Authors

BCG X — Data for Strategy Team :  
Grégoire Bidault (github : Gregoire-Bidault), Anna Silvia Saffirio, Alice Singh, Nandana Sreeraj (github : nandanasreeraj123), Hannah Hassoune-de Maximy​
