import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Cost-Optimized Churn Prediction Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Load pipeline payload
MODEL_PATH = "models/churn_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file missing. Run train.py first.")

artifact = joblib.load(MODEL_PATH)
model_pipeline = artifact["pipeline"]
OPTIMAL_THRESHOLD = artifact["optimal_threshold"]

# Add these two fields to your CustomerData class in app.py:
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    Tenure_to_Monthly_Ratio: float 
    Has_Internet: int    

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    
    # Run inference directly through the combined ColumnTransformer + Model pipeline
    prob_churn = float(model_pipeline.predict_proba(input_df)[:, 1][0])
    
    # Categorization based on optimal threshold bounds
    is_churn = prob_churn >= OPTIMAL_THRESHOLD
    if prob_churn >= 0.70:
        risk_level = "CRITICAL RISK"
        badge_color = "#dc2626"
    elif prob_churn >= OPTIMAL_THRESHOLD:
        risk_level = "HIGH RISK"
        badge_color = "#ea580c"
    elif prob_churn >= 0.25:
        risk_level = "MODERATE RISK"
        badge_color = "#ca8a04"
    else:
        risk_level = "LOW RISK"
        badge_color = "#16a34a"

    return JSONResponse({
        "churn_probability": round(prob_churn * 100, 2),
        "will_churn": bool(is_churn),
        "risk_level": risk_level,
        "badge_color": badge_color,
        "operating_threshold": round(OPTIMAL_THRESHOLD * 100, 2),
        "actionable_insight": (
            "Deploy retention coupon immediately." if is_churn 
            else "Standard operational status. No intervention required."
        )
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)