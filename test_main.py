import pytest
from model_pipeline import prepare_data, train_model, evaluate_model


def test_prepare_data():
    X_train, X_test, y_train, y_test, scaler = prepare_data("churn-bigml-80.csv")
    assert X_train is not None
    assert X_test is not None


def test_train_model():
    X_train, _, y_train, _, _ = prepare_data("churn-bigml-80.csv")
    model = train_model(X_train, y_train, n_neighbors=5)
    assert model is not None


def test_evaluate_model():
    X_train, X_test, y_train, y_test, _ = prepare_data("churn-bigml-80.csv")
    model = train_model(X_train, y_train, n_neighbors=5)
    accuracy, report = evaluate_model(model, X_test, y_test)
    assert accuracy > 0
    assert report is not None
