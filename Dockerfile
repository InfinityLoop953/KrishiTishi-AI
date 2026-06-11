# Use a stable, lightweight Python runtime
FROM python:3.12.3-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by C-based ML libraries (like XGBoost or scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching layer
COPY requirements.txt .

# Install dependencies without caching the installation files (keeps image slim)
# Fix: Force installation of tf_keras alongside the requirements file
RUN pip install --no-cache-dir -r requirements.txt tf_keras

# Copy the entire project workspace into the container's working directory (/app)
COPY . .

# Expose the default port Render uses for Web Services
EXPOSE 10000

# CRITICAL: Tell Python to treat /app as a root directory for absolute imports
ENV PYTHONPATH=/app

# Start Uvicorn pointing directly to your lifespan-managed app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]