# Enterprise Customer Churn Prediction Engine

> An end-to-end, cost-aware machine learning system for predicting telecom customer churn and prioritizing high-risk customers for retention.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-UI-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Overview

Customer retention is one of the most important challenges in the telecommunications industry.

Instead of simply predicting whether a customer will churn, this project builds a complete **production-style machine learning pipeline** that:

- Cleans and validates customer data
- Performs exploratory data analysis
- Creates meaningful customer features
- Handles numerical and categorical variables
- Trains and compares multiple ML models
- Performs cross-validation and hyperparameter tuning
- Evaluates models using business-relevant metrics
- Uses SHAP for model explainability
- Optimizes the prediction threshold based on retention costs
- Serializes the complete ML pipeline
- Provides a real-time web interface through FastAPI

The goal is to move from:
```text
Raw Customer Data
        ↓
Machine Learning Model
        ↓
Churn Prediction
```
## Machine Learning Pipeline :
```text
                    Customer Dataset
                           │
                           ▼
                  ┌─────────────────┐
                  │  Data Cleaning  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      EDA        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │Feature Engineering│
                  └────────┬────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train / Test Split   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Preprocessing      │
                │                      │
                │ Numerical → Scaling  │
                │ Categorical → OneHot │
                └──────────┬───────────┘
                           │
                           ▼
          ┌───────────────────────────────────┐
          │          Model Training           │
          │                                   │
          │ Logistic Regression               │
          │ Decision Tree                     │
          │ Random Forest                     │
          │ XGBoost                           │
          └────────────────┬──────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Cross Validation     │
                │ + Hyperparameter     │
                │   Tuning             │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation     │
                │                      │
                │ Accuracy             │
                │ Precision            │
                │ Recall               │
                │ F1 Score             │
                │ ROC-AUC              │
                │ Confusion Matrix     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ SHAP Explainability  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Cost-Based Threshold │
                │ Optimization         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Serialized Pipeline  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │     FastAPI API      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Web Dashboard      │
                └──────────────────────┘
```
## Dataset :

This project uses the IBM Telco Customer Churn Dataset.
The dataset contains customer information such as:
Customer demographics
Tenure
Contract type
Payment method
Internet service
Monthly charges
Total charges
Additional services
Churn status


## Models:

The system evaluates multiple classification algorithms.
```text
Model	Purpose
Logistic Regression	Interpretable baseline
Decision Tree	Non-linear decision rules
Random Forest	Ensemble learning
XGBoost	Gradient boosting
```
## Model Evaluation:
```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```
# Tech Stack

## 🧠 Machine Learning & Data Science

<p align="left">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>

<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn"/>

<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>

<img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>

<img src="https://img.shields.io/badge/SHAP-8A2BE2?style=for-the-badge&logo=python&logoColor=white" alt="SHAP"/>

<img src="https://img.shields.io/badge/Joblib-333333?style=for-the-badge&logo=python&logoColor=white" alt="Joblib"/>

</p>

| Technology | Purpose |
|------------|---------|
| 🐍 **Python** | Core programming language |
| 🐼 **Pandas** | Data manipulation and analysis |
| 🔢 **NumPy** | Numerical computation |
| 📊 **Matplotlib** | Data visualization |
| 🎨 **Seaborn** | Statistical visualization |
| 🤖 **Scikit-learn** | ML models, preprocessing, pipelines, cross-validation and tuning |
| 🚀 **XGBoost** | Gradient boosting classification |
| 🔍 **SHAP** | Model explainability |
| 💾 **Joblib** | ML pipeline serialization |


## ⚡ Backend & API

<p align="left">

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>

<img src="https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white" alt="Uvicorn"/>

<img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic"/>

</p>

| Technology | Purpose |
|------------|---------|
| ⚡ **FastAPI** | REST API and real-time ML inference |
| 🚀 **Uvicorn** | ASGI server for running the FastAPI application |
| ✅ **Pydantic** | Request validation and data schemas |


## 🎨 Frontend

<p align="left">

<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5"/>

<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3"/>

<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript"/>

</p>

| Technology | Purpose |
|------------|---------|
| 🌐 **HTML5** | Web page structure |
| 🎨 **CSS3** | Dashboard styling and responsive UI |
| ⚡ **JavaScript** | Dynamic frontend interactions |
| 🔗 **Fetch API** | Communication between frontend and FastAPI backend |


## System Architecture

```text
                     User
                      │
                      ▼
              HTML / CSS / JS
                      │
                      │ Fetch API
                      ▼
                FastAPI
                      │
                      ▼
                  Pydantic
                  Validation
                      │
                      ▼
                  ML Pipeline
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
      Logistic     Random       XGBoost
     Regression    Forest
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                    SHAP
              Explainability
                      │
                      ▼
                Churn Probability
                      │
                      ▼
               Risk Prediction
```    
## Repository Structure :
```text
├── data/
│   └── telco_churn.csv              # Raw Kaggle Dataset
├── models/
│   └── churn_pipeline.joblib        # Serialized ML Pipeline (Generated)
├── static/
│   ├── index.html                   # Dashboard UI
│   ├── style.css                    # Glassmorphism styling
│   └── script.js                    # Dynamic DOM handling & API requests
├── app.py                           # FastAPI Server & Inference Endpoint
├── train.py                         # End-to-End Training & Evaluation Script
└── requirements.txt                 # Project Dependencies
```
## Connect :
Built by Rajendra Kumar Ojha – Open to software engineering and AI/ML opportunities.
[[LinkedIn Profile]](https://www.linkedin.com/in/rajendra-kumar-ojha-45b47830b/)
