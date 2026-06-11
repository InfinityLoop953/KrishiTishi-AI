# Step 1: Use the official slim Python 3.12.3 image
FROM python:3.12.3-slim

# Step 2: Set the workspace directory inside the container
WORKDIR /app

# Step 3: Install system build dependencies required for your models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements first to take advantage of Docker caching
COPY requirements.txt .

# Step 5: Upgrade pip and install tensorflow-cpu/keras to save massive RAM on Render
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow-cpu==2.20.0 keras==3.14.1 && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your backend folder and root main.py into the container
COPY backend/ ./backend/
COPY main.py .

# Step 7: Expose the port Render expects web applications to use
EXPOSE 10000

# Step 8: Start your FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

