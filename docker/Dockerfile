# Base image with CUDA 12.8.0, CUDNN 8, on Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Set DEBIAN_FRONTEND to noninteractive to prevent apt-get from prompting
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# If not, Python 3.13 would need to be compiled from source.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3.12-dev \
    gcc \
    g++ \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ca-certificates && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
ENV VENV_PATH=/app/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Create working directory
WORKDIR /app

RUN python3 -m venv ${VENV_PATH}

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt ./

#
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir -r requirements.txt

# Expose Flask port (if your app.py uses Flask on this port)
EXPOSE 5001

# Start the main application
# This will use the python from the virtual environment due to the PATH modification
CMD ["python", "app.py"]