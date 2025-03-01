install:
	venv/bin/pip install -r requirements.txt

test: prepare train evaluate
push: prepare train MLflow

check:
	black . && flake8 . && bandit -r . && mypy --ignore-missing-imports .

prepare:
	python main.py prepare --train churn-bigml-80.csv --test churn-bigml-20.csv

train:
	python main.py train --train churn-bigml-80.csv --test churn-bigml-20.csv

evaluate:
	python main.py evaluate

MLflow:
	python main.py MLflow

save:
	python main.py save --save model.joblib

run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

all: install test check

