import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from libs.point import TitanicFeatures
from libs.model import predict, train

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting Titanic survival using a machine learning model.",)

@app.get("/", tags=["intro"])
async def index():
    return {"message": "Titanic Survival Prediction ML API is running"}

@app.post("/model/predict", tags=["prediction"], status_code=200)
async def get_prediction(data: TitanicFeatures):
    model_name = "model" 
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.h5")

    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}.h5' not found in the ml_models folder")
    
    features_list = [
        data.pclass,
        data.age,
        data.sibsp,
        data.parch,
        data.fare,
        data.embarked,
        data.sex,
    ]

    try:
        survival_proba = predict(features=features_list, ml_model=model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction process: {e}")

    survival_status = 1 if survival_proba > 0.5 else 0
    confidence_percent = round(float(survival_proba) * 100, 2)
    
    return {
        "status": survival_status,
        "prediction_message": "Survived" if survival_status == 1 else "Did not survive",
        "confidence": confidence_percent,
        "input_data": data.model_dump()
    }

@app.post("/model/train", tags=["model"], status_code=200)
async def train_model(data: TitanicFeatures):
    model_name = "model" 
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.h5")

    features_list = [data.pclass, data.age, data.sibsp, data.parch, data.fare, data.embarked, data.sex]
    
    try:
        train(features=features_list, ml_model=model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model saving: {e}")

    response_object = {"model_fit": "OK", "model_save": "OK"}
    return response_object

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)