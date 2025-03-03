import argparse
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
import joblib
import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore
from fastapi import FastAPI # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore
import mlflow   # type: ignore
MLFLOW_TRACKING_URI = "http://172.17.81.48:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("XGBoost Experiment")
from elasticsearch import Elasticsearch # type: ignore
import mlflow # type: ignore
es = Elasticsearch(["http://localhost:9200"])
ES_INDEX = "mlflow-metrics"


app = FastAPI()
from pydantic import BaseModel
from typing import List

class ModelInput(BaseModel):
    feature_vector: List[float]  # Ensure feature vector contains numeric values

import os
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch

# Initialize FastAPI
app = FastAPI()

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/predictions_db")

# Elasticsearch setup
es = Elasticsearch(["http://localhost:9200"])
ES_INDEX = "mlflow-metrics"

# Define the ModelInput Pydantic model
class ModelInput(BaseModel):
    feature_vector: List[float]  # Ensure feature vector contains numeric values

# DB connection function
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

@app.post("/predict")
def predict(data: ModelInput):
    try:
        # Load the trained model
        model = joblib.load("trained_model.pkl")
    except FileNotFoundError:
        return {"error": "Model not found. Train it first!"}

    # Perform prediction
    prediction = model.predict([data.feature_vector])

    # Store the prediction result in the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (features, prediction) VALUES (%s, %s)",
        (str(data.feature_vector), int(prediction[0]))
    )
    conn.commit()
    cursor.close()
    conn.close()

    # Log to Elasticsearch
    log_entry = {
        "prediction": int(prediction[0]),
        "features": data.feature_vector
    }
    es.index(index=ES_INDEX, body=log_entry)

    return {"prediction": int(prediction[0])}

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}
@app.post("/predict")
def predict(data: ModelInput):
    import joblib
    try:
        model = joblib.load("trained_model.pkl")  # Load trained model
    except FileNotFoundError:
        return {"error": "Model not found. Train it first!"}

    # Convert input to model format
    prediction = model.predict([data.feature_vector])
    return {"prediction": int(prediction[0])}  # Convert output to integer if needed

def execute_command(
    command, train_path=None, test_path=None, model_path=None, save_path=None
):
    global X_train, X_test, y_train, y_test, scaler, label_encoders, model
    if command == "load_data":
        print("\n🔄 Chargement des données...")
        train_data, test_data = load_data(train_path, test_path)
        print("\n✅ Données chargées avec succès")
        return train_data, test_data

    elif command == "prepare":
        print("\n🔄 Chargement des données...")
        train_data, test_data = load_data(train_path, test_path)
        print("\n⚙️ Préparation des données...")
        X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(
            train_data, test_data
        )
        print("\n✅ Données préparées avec succès")

    elif command == "train":
        try:
            data = joblib.load("prepared_data.pkl")
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            scaler = data["scaler"]
            label_encoders = data["label_encoders"]
        except FileNotFoundError:
            print(
                "\n⚠️ Les données préparées n'ont pas été trouvées. Exécutez 'prepare' d'abord."
            )
            return

        n_neighbors = 5
        print(f"\n🎯 Entraînement du modèle avec n_neighbors={n_neighbors}...")
        model = train_model(X_train, y_train, n_neighbors)
        print("\n✅ Modèle entraîné avec succès")
        joblib.dump(model, "trained_model.pkl")
        print("\n✅ Modèle sauvegardé sous 'trained_model.pkl'")

    elif command == "evaluate":
        try:
            model = joblib.load("trained_model.pkl")
        except FileNotFoundError:
            print("\n⚠️ Le modèle n'a pas été trouvé. Exécutez 'train' d'abord.")
            return

        try:
            # Load prepared data
            data = joblib.load("prepared_data.pkl")
            X_test = data["X_test"]
            y_test = data["y_test"]
        except FileNotFoundError:
            print(
                "\n⚠️ Les données préparées n'ont pas été trouvées. Exécutez 'prepare' d'abord."
            )
            return

        print("\n📊 Évaluation du modèle...")

        accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)

        print(f"\n✅ Précision: {accuracy}")
        print(f"🎯 Précision: {precision}")
        print(f"🔄 Rappel: {recall}")
        print(f"📊 F1-score: {f1}")
        print("\n📝 Classification Report:\n", report)

    elif command == "MLflow":
        try:
            model = joblib.load("trained_model.pkl")
        except FileNotFoundError:
            print("\n⚠️ Le modèle n'a pas été trouvé. Exécutez 'train' d'abord.")
            return

        try:
            data = joblib.load("prepared_data.pkl")
            X_test = data["X_test"]
            y_test = data["y_test"]
        except FileNotFoundError:
            print("\n⚠️ Les données préparées n'ont pas été trouvées. Exécutez 'prepare' d'abord.")
            return

        print("\n📊 Évaluation du modèle...")

        with mlflow.start_run():  # Start an MLflow run for evaluation
            accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)

            print(f"\n✅ Précision: {accuracy}")
            print(f"🎯 Précision: {precision}")
            print(f"🔄 Rappel: {recall}")
            print(f"📊 F1-score: {f1}")
            print("\n📝 Classification Report:\n", report)

            # Log metrics to MLflow
            mlflow.log_param("n_neighbors", model.n_neighbors)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1-score", f1)

            # Fix MLflow Warning: Add input example and signature
            from mlflow.models.signature import infer_signature
            import pandas as pd

            input_example = pd.DataFrame(
                X_test[:1], columns=[f"feature_{i}" for i in range(X_test.shape[1])]
            )
            signature = infer_signature(X_test, model.predict(X_test))

            mlflow.sklearn.log_model(
                model, "model", input_example=input_example, signature=signature
            )

            # --- Send Metrics to Elasticsearch ---
            log_entry = {
                "experiment": "XGBoost Experiment",
                "model": "KNN",
                "n_neighbors": model.n_neighbors,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            }

            es.index(index=ES_INDEX, body=log_entry)  # Send data to Elasticsearch
            print("\n✅ Metrics logged to Elasticsearch")


    elif command == "save":
        if save_path:
            try:
                if "model" not in globals():
                    model = joblib.load("trained_model.pkl")

                data = joblib.load("prepared_data.pkl")
                scaler = data.get("scaler")
                label_encoders = data.get("label_encoders")

                print("\n💾 Sauvegarde du modèle...")
                joblib.dump(
                    {
                        "model": model,
                        "scaler": scaler,
                        "label_encoders": label_encoders,
                    },
                    save_path,
                )
                print(f"\n✅ Modèle sauvegardé sous {save_path}")
            except FileNotFoundError:
                print(
                    "\n⚠️ Impossible de sauvegarder: le modèle ou les données préparées sont introuvables."
                )
        else:
            print("\n⚠️ Spécifiez un chemin pour la sauvegarde avec --save")

    elif command == "load":
        if model_path:
            print("\n📥 Chargement du modèle sauvegardé...")
            model_data = load_model(model_path)
            print("\n✅ Modèle chargé avec succès")
        else:
            print("\n⚠️ Spécifiez un chemin pour charger un modèle avec --load")

    else:
        print("\n❌ Commande invalide. Utilisez prepare, train, evaluate, save, load")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de traitement de données et apprentissage automatique"
    )
    parser.add_argument(
        "command",
        type=str,
        help="Commande à exécuter: load_data, prepare, train, evaluate, save, load",
    )
    parser.add_argument(
        "--train", type=str, help="Chemin vers le fichier d'entraînement"
    )
    parser.add_argument("--test", type=str, help="Chemin vers le fichier de test")
    parser.add_argument("--load", type=str, help="Chemin vers un modèle sauvegardé")
    parser.add_argument("--save", type=str, help="Chemin pour sauvegarder le modèle")
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        execute_command(
            args.command,
            train_path=args.train,
            test_path=args.test,
            model_path=args.load,
            save_path=args.save,
        )
