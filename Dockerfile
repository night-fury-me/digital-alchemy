# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;8.6"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install required Python dependencies
RUN pip install --no-cache-dir \
    pytorch-lightning>=1.9.0 \
    hydra-core>=1.1.0 \
    ase>=3.21 \
    schnetpack \
    mlflow \
    torchmetrics \
    matplotlib

# Expose MLflow UI port
EXPOSE 5000

# Set the working directory
WORKDIR /workspace

# Ensure MLflow logs persist
VOLUME /workspace/mlruns

# Start MLflow tracking server in background, then run bash
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/workspace/mlruns"]
