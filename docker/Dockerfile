FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu22.04

# 安裝 Python 與必要工具（無快取、減少體積）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 設定 python 為預設版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 建立工作目錄
WORKDIR /app

# 只安裝依賴，不 COPY 原始碼
COPY requirements.txt ./

# 安裝 CUDA 12.8 對應的 PyTorch + 其他依賴
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt

# 開放 Flask port
EXPOSE 5001

# 啟動主程式
CMD ["python", "app.py"]
