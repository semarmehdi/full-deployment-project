import os
import mlflow
import pandas as pd
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Union
from dot_env import load_dotenv


load_dotenv()

# -----------------------------------------------------------------------------
# ENV + MLflow setup
# -----------------------------------------------------------------------------
# Sur Hugging Face, ces variables sont lues depuis les "Secrets"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS")

# On force l'URI pour mlflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def build_model_uri() -> str:
    if MODEL_ALIAS:
        return f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    return f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE or 'Production'}"
    # models:/ibm_attrition_detector@production


MODEL_URI = build_model_uri()
MODEL = None

# -----------------------------------------------------------------------------
# FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="👨‍💼 HR Prediction API")


class PredictionFeatures(BaseModel):
    Age: Union[int, float]
    BusinessTravel: str
    DailyRate: Union[int, float]
    Department: str
    DistanceFromHome: Union[int, float]
    Education: Union[int, float]
    EducationField: str
    EmployeeCount: Union[int, float]
    EmployeeNumber: Union[int, float]
    EnvironmentSatisfaction: Union[int, float]
    Gender: str
    HourlyRate: Union[int, float]
    JobInvolvement: Union[int, float]
    JobLevel: Union[int, float]
    JobRole: str
    JobSatisfaction: Union[int, float]
    MaritalStatus: str
    MonthlyIncome: Union[int, float]
    MonthlyRate: Union[int, float]
    NumCompaniesWorked: Union[int, float]
    Over18: str
    OverTime: str
    PercentSalaryHike: Union[int, float]
    PerformanceRating: Union[int, float]
    RelationshipSatisfaction: Union[int, float]
    StandardHours: Union[int, float]
    StockOptionLevel: Union[int, float]
    TotalWorkingYears: Union[int, float]
    TrainingTimesLastYear: Union[int, float]
    WorkLifeBalance: Union[int, float]
    YearsAtCompany: Union[int, float]
    YearsInCurrentRole: Union[int, float]
    YearsSinceLastPromotion: Union[int, float]
    YearsWithCurrManager: Union[int, float]


# -----------------------------------------------------------------------------
# Startup: CHARGEMENT BLOQUANT (Solution au bug 500)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_model_sync():
    global MODEL
    print(f"🚀 [INFO] Attempting to load model: {MODEL_URI}")
    try:
        # On attend que le chargement soit fini avant de rendre l'API disponible
        MODEL = mlflow.sklearn.load_model(MODEL_URI)
        # models:/ibm_attrition_detector@production
        print("✅ [INFO] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        # En cas d'échec, on laisse MODEL à None pour que /health le signale


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
async def predict(payload: PredictionFeatures):
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="Model is still loading or failed to load."
        )

    # Conversion pydantic -> dict -> DataFrame
    df = pd.DataFrame([payload.dict()])
    pred = MODEL.predict(df)
    return {"prediction": int(pred[0])}


if __name__ == "__main__":
    # Port 7860 est le standard pour Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
