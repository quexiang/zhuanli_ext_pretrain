# 使用 NVIDIA CUDA 11.8 作为基础镜像（兼容 PaddlePaddle GPU）
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 设置环境变量，防止交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 【关键修复】替换为清华 Ubuntu 源（解决 apt-get 连接超时）
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖（PaddleOCR 所需的图像处理库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制依赖文件并安装 Python 包（使用清华源加速）
COPY requirements.txt .
RUN pip3 install paddlepaddle-gpu==3.0.0 paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目所有代码
COPY app/ ./app/

# 创建运行时需要的文件夹
RUN mkdir -p uploads outputs

# 暴露端口
EXPOSE 8020

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]