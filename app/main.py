import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .schemas import InputFeatures, OutputPrediction
import joblib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
app = FastAPI()
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

rf_model = joblib.load('app/model/random_forest_model.joblib')
scaler = joblib.load('app/model/scaler.joblib')

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")
from fastapi.responses import FileResponse

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")


@app.get("/info")
def get_info():
    return {"message": "Heart Disease Prediction API", "version": "1.0"}

@app.post("/predict", response_model=OutputPrediction)
def predict_heart_disease(input: InputFeatures):
    try:
        features = input.dict()
        input_data = pd.DataFrame([features])
        input_data = input_data[feature_names]  # ensure correct order
        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)
        probability = rf_model.predict_proba(input_scaled)[0][1]
        return OutputPrediction(prediction=int(prediction[0]), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
