from fastapi import FastAPI, UploadFile, File
import pandas as pd
from src.models import train_model, predict_model

app = FastAPI(title="ML Training & Prediction API")

# Train endpoint
@app.post("/train")
def train_model_api():
    best_model, results, cv_results = train_model.main()
    return {
        "message": "Training completed successfully",
        "best_model": str(best_model),
        "results": {k: v["accuracy"] for k, v in results.items()},
    }

# Predict endpoint
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        # Load trained model
        model = predict_model.load_trained_model("best_model.pkl")
        if model is None:
            return {"error": "No trained model found. Please train first."}

        # Read uploaded file
        df = pd.read_csv(file.file)

        predictions, probabilities = predict_model.predict_new_data(model, df)
        summary = predict_model.create_prediction_summary(predictions, probabilities)

        return summary.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}