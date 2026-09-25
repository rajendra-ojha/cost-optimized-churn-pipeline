import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
    print("Loading Real Kaggle Dataset...")
    data_path = "data/telco_churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Cannot find {data_path}. Download from Kaggle first.")
    
    df = pd.read_csv(data_path)
    
    print("Engineering Features & Cleaning Data...")
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Advanced Feature Engineering (Recruiters look for this)
    df["Tenure_to_Monthly_Ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1)
    df["Has_Internet"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)
    
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # Handling Class Imbalance (Crucial for high Recall/AUC)
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            "params": {"classifier__C": [0.1, 1.0, 10.0]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {"classifier__n_estimators": [200], "classifier__max_depth": [8, 12]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", scale_pos_weight=3, random_state=42),
            "params": {"classifier__n_estimators": [100, 200], "classifier__learning_rate": [0.05, 0.1]}
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_score = -1
    best_pipeline = None

    print(f"\n{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 75)

    for name, config in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", config["model"])])
        search = GridSearchCV(pipe, config["params"], cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f} | {auc:.4f}")

        if auc > best_overall_score:
            best_overall_score = auc
            best_pipeline = best

    print("\nSaving Best Production Model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "optimal_threshold": 0.45}, "models/churn_pipeline.joblib")
    print("Pipeline successfully saved to 'models/churn_pipeline.joblib'.")

if __name__ == "__main__":
    main()