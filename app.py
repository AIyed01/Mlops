from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Charger le modèle sauvegardé
try:
    model_data = joblib.load("trained_model.pkl")
    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
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

class RetrainParams(BaseModel):
    n_neighbors: int

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        if features.shape[1] != model.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Le modèle attend {model.n_features_in_} features, mais {features.shape[1]} ont été fournis.")
        
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
def retrain(params: RetrainParams):
    try:
        # Charger les données préparées
        data = joblib.load('prepared_data.pkl')
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        # Réentraîner le modèle avec les nouveaux hyperparamètres
        new_model = KNeighborsClassifier(n_neighbors=params.n_neighbors)
        new_model.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_pred = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Sauvegarder le nouveau modèle
        joblib.dump({"model": new_model, "scaler": scaler, "label_encoders": label_encoders}, "trained_model.pkl")
        
        return {"message": "Modèle réentraîné avec succès", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
