# enterprise-customer-churn-prediction
## Overview

A production-grade machine learning platform for predicting customer churn using advanced machine learning, explainable AI, MLOps practices, and cloud deployment.

## Objectives

* Predict customer churn probability
* Identify churn drivers
* Provide retention recommendations
* Deploy a scalable prediction API
* Build executive analytics dashboards

## Technology Stack

* Python
* Scikit-Learn
* XGBoost
* PostgreSQL
* FastAPI
* Streamlit
* Docker
* AWS
* MLflow
* DVC
* Evidently AI

file structure:--- 
 enterprise-customer-churn-prediction:---
│
├── configs/           # Settings in YAML: paths, schema, hyperparameters.
│                      #   Keeps values OUT of code so you change one file, not ten.
│
├── data/              # All datasets, separated by HOW processed they are
│   ├── raw/           #   Original data, downloaded once, NEVER edited by hand
│   ├── interim/       #   Half-cleaned / intermediate steps
│   └── processed/     #   Final, model-ready tables
│
├── notebook/          # Jupyter notebooks for EDA + experiments (messy exploring lives here)
│
├── src/               # The real, reusable pipeline code — ONE folder per stage
│   ├── ingestion/     #   Load raw data + check it's valid     <-- YOU ARE HERE
│   ├── preprocessing/ #   Clean, fix missing values, encode categories
│   ├── features/      #   Engineer new predictive columns
│   ├── training/      #   Train the model(s)
│   ├── evaluation/    #   Measure performance + explainability (SHAP)
│   └── validation/    #   Data-quality / model sanity checks (gates before trusting output)
│
├── models/            # Saved trained models (.pkl / .joblib files)
│
├── reports/           # Generated outputs: charts, metrics, dashboards
│
├── tests/             # Unit tests for the code in src/
│
├── venv/              # Your virtual environment (gitignored — never committed)
├── .gitignore         # Tells git what to skip (venv/, large data, models)
├── main.py            # Entry point — runs the stages in order
├── requirements.txt   # List of pip packages the project needs
└── README.md          # Project documentation
