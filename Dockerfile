# Use a Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (this allows passing arguments like "serve")
ENTRYPOINT ["python", "main.py"]

