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
from transformers import AutoModelForImageSegmentation, SegformerForSemanticSegmentation, SegformerImageProcessor, AutoImageProcessor, AutoModelForDepthEstimation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True); \
SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'); \
SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'); \
AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
print('segmentation + depth weights cached')"

# Guard: HF_HUB_OFFLINE=1 below means anything missed here can never load at
# runtime -- it fails silently into "masking failed, continuing without
# regions" and the affected recipes just never appear. Depth Anything was
# added to masking.py without being added here, and that is exactly how it
# behaved in production. Fail the build instead.
RUN python -c "\
import os; os.environ['HF_HUB_OFFLINE'] = '1'; \
from transformers import AutoModelForImageSegmentation, SegformerForSemanticSegmentation, AutoModelForDepthEstimation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True); \
SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'); \
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
print('verified: every model loads with the Hub switched off')"

# Models are baked above, so don't reach out to the Hub at runtime -- a
# network hiccup should never turn into a failed import.
ENV HF_HUB_OFFLINE=1

WORKDIR /app/service
ENV PYTHONUNBUFFERED=1
EXPOSE 5054

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5054"]
