FROM python:3.12-slim

# Install system libraries required by PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install PaddlePaddle GPU version (cu118 works on CUDA 12.x)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.1

# Install application dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create runtime directories
RUN mkdir -p uploads outputs

EXPOSE 8020

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]
