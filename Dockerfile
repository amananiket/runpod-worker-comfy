# Stage 1: Base image with common dependencies
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

ADD src/extra_model_paths.yaml ./

ADD ["assets/Body Type_alpha1.0_rank4_noxattn_last.safetensors", "./models/loras/"]

# Install runpod
RUN pip3 install runpod requests

RUN pip3 install -r requirements.txt

WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py src/garment_segmentation_model.py ./
ADD src/segformer/ ./segformer/

RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy snapshot file
ADD snapshot.jso[n] /

RUN /restore_snapshot.sh

# Start the container
CMD /start.sh