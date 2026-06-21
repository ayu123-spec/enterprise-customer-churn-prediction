# ---- Enterprise Customer Churn Prediction API ----
FROM python:3.12-slim

WORKDIR /app

# Install only the dependencies needed to SERVE the model (keeps the image small)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API code and the trained model artifacts
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Serve the FastAPI app, listening on all interfaces inside the container
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
