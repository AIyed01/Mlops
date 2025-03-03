# Variables
IMAGE_NAME = votre_prenom_votre_nom_vote_classe_mlops
DOCKER_USERNAME = votre_utilisateur_dockerhub

# Install dependencies
install:
	venv/bin/pip install -r requirements.txt

# Test the code with linting and type checks
test: prepare train evaluate
	push: prepare train MLflow

check:
	black . && flake8 . && bandit -r . && mypy --ignore-missing-imports .

# Prepare the data for training
prepare:
	python main.py prepare --train churn-bigml-80.csv --test churn-bigml-20.csv

# Train the model
train:
	python main.py train --train churn-bigml-80.csv --test churn-bigml-20.csv

# Evaluate the model
evaluate:
	python main.py evaluate

# Run MLflow
MLflow:
	python main.py MLflow

# Save the model
save:
	python main.py save --save model.joblib

# Run the API with FastAPI
run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000


# Build the Docker image
build:
	docker build -t iyed_ayadi_4ds7_mlops .

# Tag the Docker image
tag:
	docker tag iyed_ayadi_4ds7_mlops aiyed01/iyed_ayadi_4ds7_mlops

# Push the Docker image to Docker Hub
push:
	docker push aiyed01/iyed_ayadi_4ds7_mlops

# Run the Docker container locally
run:
	docker run -p 8000:8000 aiyed01/iyed_ayadi_4ds7_mlops

# Remove unused Docker images
clean:
	docker image prune -f

# Run all tasks: install dependencies, run tests, and check the code
all: install test check
