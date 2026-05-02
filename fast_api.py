from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model=joblib.load("depresion_lebel.pkl")

try:
    scaler=joblib.load("scaler.pkl")
except:
    scaler=None




app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

class Inputdata(BaseModel):
    sleep_hours: float
    daily_social_media_hours: float
    stress_level: float
    anxiety_level: float
    social_interaction_level: float
    academic_performance: float
    physical_activity: float
    screen_time_before_sleep: float
    addiction_level: float

@app.get("/")
def home():
        return{"message":"Depression Prediction API Running"}
@app.post("/predict")
def predict(data: Inputdata):
    try:
        input_data = np.array([
            data.sleep_hours,
            data.daily_social_media_hours,
            data.stress_level,
            data.anxiety_level,
            data.social_interaction_level,
            data.academic_performance,
            data.physical_activity,
            data.screen_time_before_sleep,
            data.addiction_level
        ]).reshape(1, -1)

        print("Input:", input_data)

        if scaler:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        return {
            "prediction": int(prediction),
            "Staus": "Depression" if prediction == 1 else "Not Depressed",
            "Health":"Good" if prediction == 0 else "Not Good"}
    except Exception as e:
        return {"error": str(e)}