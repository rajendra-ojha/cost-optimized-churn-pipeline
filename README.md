# 🚀 Enterprise Customer Churn Prediction Engine

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

## 📌 Overview

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
