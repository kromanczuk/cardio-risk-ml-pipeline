"""Pydantic request/response models for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 2,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 7,
}


class PatientFeatures(BaseModel):
    """The 13 clinical features from UCI Heart Disease, with the original UCI encodings."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain: 1 = typical angina, 2 = atypical angina, "
        "3 = non-anginal pain, 4 = asymptomatic"
    )
    trestbps: int = Field(ge=80, le=220, description="Resting blood pressure (mm Hg)")
    chol: int = Field(ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl: 1 = true")
    restecg: Literal[0, 1, 2] = Field(
        description="Resting ECG: 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy"
    )
    thalach: int = Field(ge=60, le=220, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina: 1 = yes")
    oldpeak: float = Field(ge=0, le=7, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        description="Slope of peak exercise ST segment: 1 = up, 2 = flat, 3 = down"
    )
    ca: int = Field(ge=0, le=3, description="Number of major vessels colored by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(
        description="Thalassemia: 3 = normal, 6 = fixed defect, 7 = reversible defect"
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "probability": 0.73,
                    "prediction": 1,
                    "risk_level": "high",
                    "model_version": "20261120-153000-a1b2c3",
                }
            ]
        }
    )

    probability: float = Field(ge=0, le=1, description="Predicted probability of heart disease")
    prediction: Literal[0, 1] = Field(description="1 if probability ≥ 0.5, else 0")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="low < 0.3 ≤ medium < 0.6 ≤ high"
    )
    model_version: str = Field(description="Version of the model that made this prediction")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "ok", "model_loaded": True, "model_version": "20261120-153000-a1b2c3"},
                {"status": "ok", "model_loaded": False, "model_version": None},
            ]
        }
    )

    status: Literal["ok"] = Field(description="API is running")
    model_loaded: bool = Field(description="Whether a model was loaded at startup")
    model_version: str | None = Field(default=None, description="Loaded model's version, if any")
