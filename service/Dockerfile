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

# Compile BiRefNet to an OpenVINO IR at build time.
#
# BiRefNet's input is always 1x3x1024x1024, and with that shape pinned
# OpenVINO runs it 5.4x faster than PyTorch on the same CPU (12.00s -> 2.20s
# on optiplex), producing masks identical to 2.9e-08. It is 32% of an import,
# so this is the single largest saving in the pipeline.
#
# Built here rather than on first use because conversion takes ~70s, which
# would otherwise be paid by whoever imports the first photo after a deploy.
# convert_model leaves the spatial dims dynamic; reshape pins them, which the
# GPU plugin requires outright and the CPU plugin needs for its best kernels.
RUN python -c "\
import torch, openvino as ov; \
from transformers import AutoModelForImageSegmentation; \
m=AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True).eval(); \
ovm=ov.convert_model(m, example_input=torch.zeros(1,3,1024,1024)); \
ovm.reshape({ovm.input(0): ov.PartialShape([1,3,1024,1024])}); \
ov.save_model(ovm, '/app/service/weights/birefnet_lite.xml'); \
import os; \
print('birefnet IR', os.path.getsize('/app/service/weights/birefnet_lite.bin')//1024//1024, 'MB')"

# SCUNet denoise weights (the authors' real-noise model). Not a
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

# FFDNet: 0.85M parameters against SCUNet's 17.9M, and it works on a 2x2
# pixel-shuffled sub-image, so a 26MP frame costs 41.6s rather than 476s on
# this hardware. 3.4MB, so the same fetch-and-verify as above but cheap.
RUN python -c "\
import hashlib, os, sys, urllib.request; \
url='https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth'; \
dst='/app/service/weights/ffdnet_color.pth'; \
os.makedirs(os.path.dirname(dst), exist_ok=True); \
urllib.request.urlretrieve(url, dst); \
h=hashlib.sha256(open(dst,'rb').read()).hexdigest(); \
print('ffdnet sha256', h, os.path.getsize(dst)//1024, 'KB'); \
sys.exit(0 if h.startswith('286f04f7e3a20254') else 1)"


# --- Intel iGPU compute runtime (optional acceleration for the denoiser) ---
#
# The OpenVINO wheel already ships libopenvino_intel_gpu_plugin.so, but with
# no Intel driver underneath it reports no GPU at all and silently runs on
# CPU. That is very likely why an earlier "openvino_gpu is no faster than
# CPU" benchmark came out flat -- it was comparing CPU against CPU.
#
# Two traps, both found the hard way:
#  1. This box's UHD 630 is Gen9.5, which current compute-runtime releases
#     have DROPPED. Legacy platforms need the 24.35 line, per Intel's own
#     LEGACY_PLATFORMS.md -- installing the newest driver would never see it.
#  2. intel-opencl-icd-legacy1 pins an exact IGC build (1.0.17537.24). Pairing
#     it with a newer IGC leaves the OpenCL ICD unconfigured.
#
# Measured once working, on an idle box: SCUNet 5.67s -> 3.39s per 512px
# tile, a 1.7x speedup. (An earlier 9.75 -> 3.86 / 2.52x recorded here was
# taken while the machine was busy and overstated both ends.)
ARG CR=24.35.30872.36
ARG IGC=1.0.17537.24
RUN set -eux; \
    apt-get update -qq; \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 libze1; \
    cd /tmp; \
    for f in \
      "https://github.com/intel/intel-graphics-compiler/releases/download/igc-${IGC}/intel-igc-core_${IGC}_amd64.deb" \
      "https://github.com/intel/intel-graphics-compiler/releases/download/igc-${IGC}/intel-igc-opencl_${IGC}_amd64.deb" \
      "https://github.com/intel/compute-runtime/releases/download/${CR}/libigdgmm12_22.5.0_amd64.deb" \
      "https://github.com/intel/compute-runtime/releases/download/${CR}/intel-opencl-icd-legacy1_${CR}_amd64.deb" \
      "https://github.com/intel/compute-runtime/releases/download/${CR}/intel-level-zero-gpu-legacy1_1.5.${CR#*.*.}_amd64.deb" ; \
      do python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[1].rsplit('/',1)[1])" "$f"; done; \
    dpkg -i ./*.deb || apt-get -f install -y --no-install-recommends; \
    for v in /etc/OpenCL/vendors/*.dpkg-new; do [ -e "$v" ] && mv "$v" "${v%.dpkg-new}" || true; done; \
    rm -f /tmp/*.deb; \
    rm -rf /var/lib/apt/lists/*; \
    ls /etc/OpenCL/vendors/

# Models are baked above, so don't reach out to the Hub at runtime -- a
# network hiccup should never turn into a failed import.
ENV HF_HUB_OFFLINE=1

# A missing or unloadable IR degrades to torch SILENTLY -- 5.4x slower with
# no error anywhere. That is exactly the failure mode the offline model guard
# exists to prevent, so fail the build instead.
RUN cd /app/service && python birefnet_parity.py

WORKDIR /app/service
ENV PYTHONUNBUFFERED=1
EXPOSE 5054

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5054"]
