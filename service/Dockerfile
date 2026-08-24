# Build context is the REPO ROOT (not this directory), so this Dockerfile can
# copy in shared/model.py -- see docker-compose.yml's `context`/`dockerfile`
# split in homelab-infra. Container layout mirrors the repo layout
# (/app/shared, /app/service) so service code's `sys.path`-based
# `from shared.model import ...` import works identically to running it
# straight from a checkout.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch wheel explicitly, to avoid pulling multi-GB CUDA libs this
# CPU-only host (optiplex, no discrete GPU) will never use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY shared/ ./shared/
COPY service/ ./service/

WORKDIR /app/service
ENV PYTHONUNBUFFERED=1
EXPOSE 5054

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5054"]
