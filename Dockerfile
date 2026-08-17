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

# Install application dependencies.  requirements.txt installs the GPU build
# of PaddlePaddle (paddlepaddle-gpu==3.2.0, CUDA 11.8, via the cu118 extra
# index) — do NOT install a separate paddlepaddle here or the CPU wheel would
# clobber the GPU build.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create runtime directories
RUN mkdir -p uploads outputs

EXPOSE 8021

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8021"]
