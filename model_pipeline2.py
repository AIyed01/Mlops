import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def load_data(train_path, test_path):
    """Charge les fichiers CSV et retourne les dataframes."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def prepare_data(train_data, test_data):
    """Prépare les données en encodant les variables catégorielles et en normalisant les features."""
    data = pd.concat([train_data, test_data], ignore_index=True)

    label_encoders = {}
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    features = data.drop(
        columns=["Churn"], errors="ignore"
    )  # Replace 'Churn' with your actual target column name
    target = data["Churn"]
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    # Save the prepared data
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "label_encoders": label_encoders,
        },
        "prepared_data.pkl",
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders


def train_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur l'ensemble de test et retourne toutes les métriques nécessaires."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    return accuracy, precision, recall, f1, report


def save_model(model, scaler, label_encoders, filename="model.pkl"):
    """Sauvegarde le modèle et les transformateurs."""
    joblib.dump(
        {"model": model, "scaler": scaler, "encoders": label_encoders}, filename
    )


def load_model(filename="model.pkl"):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)
