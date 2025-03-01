from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# Charger le modèle sauvegardé
try:
    model_data = joblib.load("trained_model.pkl")
    if isinstance(model_data, dict):
        model = model_data["model"]
        scaler = model_data["scaler"]
        label_encoders = model_data["label_encoders"]
    else:
        model = model_data
        scaler = None
        label_encoders = None

    print(f"Le modèle attend {model.n_features_in_} features en entrée.")
except FileNotFoundError:
    raise Exception("Modèle non trouvé. Exécutez d'abord l'entraînement.")

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    try:
        # Vérifier que le bon nombre de features est fourni
        features = np.array(data.features).reshape(1, -1)
        if features.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Le modèle attend {model.n_features_in_} features, mais {features.shape[1]} ont été fournis.",
            )

        # Appliquer la transformation si un scaler est disponible
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
