#!/usr/bin/env bash

set -e 

# Quit script if snapshot file doesn't exist

if [ ! -f /snapshot.json ]; then
    echo "runpod-worker-comfy: No snapshot file found. Exiting..."
    exit 0
fi

cd /comfyui/

# Install ComfyUI-Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager

cd custom_nodes/ComfyUI-Manager

pip install -r requirements.txt

mkdir startup-scripts
mv /snapshot.json startup-scripts/restore-snapshot.json

cd ../..

# Trigger restoring of the snapshot by performing a quick test run
# Note: We need to use `yes` as some custom nodes may try to install dependencies with pip
/usr/bin/yes | python3 main.py --cpu --quick-test-for-ci

# Install insightid dependencies

wget -O /comfyui/models/checkpoints/dreamshaperxl.safetensors "https://civitai.com/api/download/models/351306?type=Model&format=SafeTensor&size=full&fp=fp16"

mkdir -p /comfyui/models/insightface/models
mkdir -p /comfyui/models/insightface/models/antelopev2

wget -O /comfyui/models/insightface/models/antelopev2/1k3d68.onnx https://alle-static.s3.ap-south-1.amazonaws.com/antelopev2/1k3d68.onnx
wget -O /comfyui/models/insightface/models/antelopev2/2d106det.onnx https://alle-static.s3.ap-south-1.amazonaws.com/antelopev2/2d106det.onnx
wget -O /comfyui/models/insightface/models/antelopev2/genderage.onnx https://alle-static.s3.ap-south-1.amazonaws.com/antelopev2/genderage.onnx
wget -O /comfyui/models/insightface/models/antelopev2/glintr100.onnx https://alle-static.s3.ap-south-1.amazonaws.com/antelopev2/glintr100.onnx
wget -O /comfyui/models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx https://alle-static.s3.ap-south-1.amazonaws.com/antelopev2/scrfd_10g_bnkps.onnx

mkdir -p /comfyui/models/insightface/models/buffalo_l

wget -O /comfyui/models/insightface/models/buffalo_l/1k3d68.onnx https://alle-static.s3.ap-south-1.amazonaws.com/buffalo_l/1k3d68.onnx
wget -O /comfyui/models/insightface/models/buffalo_l/2d106det.onnx https://alle-static.s3.ap-south-1.amazonaws.com/buffalo_l/2d106det.onnx
wget -O /comfyui/models/insightface/models/buffalo_l/det_10g.onnx https://alle-static.s3.ap-south-1.amazonaws.com/buffalo_l/det_10g.onnx
wget -O /comfyui/models/insightface/models/buffalo_l/genderage.onnx https://alle-static.s3.ap-south-1.amazonaws.com/buffalo_l/genderage.onnx
wget -O /comfyui/models/insightface/models/buffalo_l/w600k_r50.onnx https://alle-static.s3.ap-south-1.amazonaws.com/buffalo_l/w600k_r50.onnx

mkdir -p /comfyui/models/instantid
wget -O /comfyui/models/instantid/ip-adapter.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true

mkdir -p /comfyui/models/controlnet
mkdir -p /comfyui/models/controlnet/instantid
wget -O /comfyui/models/controlnet/instantid/diffusion_pytorch_model.safetensors https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true

wget -O /comfyui/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors

mkdir -p /comfyui/models/ipadapter

wget -O /comfyui/models/ipadapter/ip-adapter-faceid-portrait_sdxl.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin
wget -O /comfyui/models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
wget -O /comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
wget -O /comfyui/models/ipadapter/ip-adapter_sdxl.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors
wget -O /comfyui/models/ipadapter/ip-adapter_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors

mkdir -p /comfyui/models/bert-base-uncased

wget -O /comfyui/models/bert-base-uncased/config.json https://alle-static.s3.ap-south-1.amazonaws.com/bert-base-uncased/config.json
wget -O /comfyui/models/bert-base-uncased/model.safetensors https://alle-static.s3.ap-south-1.amazonaws.com/bert-base-uncased/model.safetensors
wget -O /comfyui/models/bert-base-uncased/tokenizer_config.json https://alle-static.s3.ap-south-1.amazonaws.com/bert-base-uncased/tokenizer-config.json
wget -O /comfyui/models/bert-base-uncased/tokenizer.json https://alle-static.s3.ap-south-1.amazonaws.com/bert-base-uncased/tokenizer.json
wget -O /comfyui/models/bert-base-uncased/vocab.txt https://alle-static.s3.ap-south-1.amazonaws.com/bert-base-uncased/vocab.txt

mkdir -p /comfyui/models/facerestore_models

wget -O /comfyui/models/facerestore_models/GFPGANv1.3.pth https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth?download=true
wget -O /comfyui/models/facerestore_models/GFPGANv1.4.pth https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth?download=true
wget -O /comfyui/models/facerestore_models/GPEN-BFR-1024.onnx https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx?download=true
wget -O /comfyui/models/facerestore_models/GPEN-BFR-2048.onnx https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx?download=true
wget -O /comfyui/models/facerestore_models/GPEN-BFR-512.onnx https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx?download=true
wget -O /comfyui/models/facerestore_models/codeformer-v0.1.0.pth https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth?download=true

mkdir -p /comfyui/models/facedetection

wget -O /comfyui/models/facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth

wget -O /comfyui/models/facedetection/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth

mkdir -p /comfyui/models/grounding-dino

wget -O /comfyui/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py

wget -O /comfyui/models/grounding-dino/groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

mkdir -p /comfyui/models/sams

wget -O /comfyui/models/sams/sam_hq_vit_b.pth https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth

mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts

mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/LayerNorm

mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/LayerNorm/DensePose-TorchScript-with-hint-image

wget -O /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/LayerNorm/DensePose-TorchScript-with-hint-image/densepose_r50_fpn_dl.torchscript https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript
