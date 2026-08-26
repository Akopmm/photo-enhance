# Build context is the REPO ROOT (not this directory), so this Dockerfile can
# copy in shared/model.py -- see docker-compose in homelab-infra. Container
# layout mirrors the repo layout (/app/shared, /app/service) so the service's
# `from shared.model import ...` works the same as in a checkout.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch wheel explicitly: this deploys to a box with no discrete
# GPU, and the default wheel would drag in multi-GB CUDA libraries it can
# never use.
# torchvision MUST come from the same CPU index and be resolved together with
# torch -- timm pulls torchvision in transitively, and letting pip fetch it
# from the default index yields a CUDA build whose compiled ops don't match
# this torch ("operator torchvision::nms does not exist" at import time).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY shared/ ./shared/
COPY service/ ./service/

# Bake the segmentation weights into the image rather than downloading them
# on first use. Costs ~250MB of image size but makes the deployed artifact
# self-contained: no cold-start stall on the first enhanced-mode import, and
# no dependency on Hugging Face being reachable from the deploy host.
ENV HF_HOME=/opt/hf
ENV HF_HUB_OFFLINE=0
RUN python -c "\
from transformers import AutoModelForImageSegmentation, UperNetForSemanticSegmentation, AutoImageProcessor, AutoModelForDepthEstimation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True); \
AutoImageProcessor.from_pretrained('openmmlab/upernet-convnext-tiny'); \
UperNetForSemanticSegmentation.from_pretrained('openmmlab/upernet-convnext-tiny'); \
AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
print('segmentation + depth weights cached')"

# SCUNet denoise weights (Apache-2.0, the authors' real-noise model). Not a
# transformers model, so fetched directly. python:3.12-slim has no curl, and
# the checksum is verified in the same step -- a silently swapped 72MB tensor
# file would be hard to notice from the output alone.
RUN python -c "\
import hashlib, os, sys, urllib.request; \
url='https://huggingface.co/deepinv/scunet/resolve/main/scunet_color_real_psnr.pth'; \
dst='/app/service/weights/scunet_color_real_psnr.pth'; \
os.makedirs(os.path.dirname(dst), exist_ok=True); \
urllib.request.urlretrieve(url, dst); \
h=hashlib.sha256(open(dst,'rb').read()).hexdigest(); \
print('scunet sha256', h, os.path.getsize(dst)//1024, 'KB'); \
sys.exit(0 if h.startswith('fa78899ba2caec9d') else 1)"

# Models are baked above, so don't reach out to the Hub at runtime -- a
# network hiccup should never turn into a failed import.
ENV HF_HUB_OFFLINE=1

WORKDIR /app/service
ENV PYTHONUNBUFFERED=1
EXPOSE 5054

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5054"]
