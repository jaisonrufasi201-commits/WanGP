FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Build arg for GPU architectures
ARG CUDA_ARCHITECTURES="8.0;8.6"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y \
    python3 python3-pip git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch
RUN pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install SageAttention from git
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="1"

COPY <<EOF /tmp/patch_setup.py
import os
with open('setup.py', 'r') as f:
    content = f.read()

arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
arch_set = '{' + ', '.join([f'"{arch}"' for arch in arch_list.split(';')]) + '}'

old_section = '''compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")'''

new_section = 'compute_capabilities = ' + arch_set + '''
print(f"Manually set compute capabilities: {compute_capabilities}")'''

content = content.replace(old_section, new_section)

with open('setup.py', 'w') as f:
    f.write(content)
EOF

RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    python3 /tmp/patch_setup.py && \
    pip install --no-cache-dir --no-build-isolation . && \
    rm -rf /tmp/sageattention /tmp/patch_setup.py

# CRITICAL FIX: Copy all application files (including wgp.py) into /workspace
COPY . /workspace

RUN useradd -u 1000 -ms /bin/bash user

RUN chown -R user:user /workspace

RUN mkdir -p /home/user/.cache && \
    chown -R user:user /home/user/.cache

COPY entrypoint.sh /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
