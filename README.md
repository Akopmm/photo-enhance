# photo-enhance

A self-hosted Lightroom-style photo enhancement service: point it at a photo (Canon CR3, Sony ARW, or plain JPEG), it predicts a color/exposure correction with a small neural network, then renders several distinct style variants (natural, HDR, cinematic teal-orange, B&W, etc.) on top of that correction. Built to run entirely on a home server's CPU (no GPU required), with a browser UI for importing from an [Immich](https://immich.app) photo library or a direct upload.

## Why

Built as a personal alternative to an Adobe Lightroom subscription.

## How it works

- **The model** is a small CNN (~600K parameters) based on the architecture from [*Learning Image-Adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time*](https://github.com/HuiZeng/Image-Adaptive-3DLUT) (Zeng et al., Apache-2.0). It looks at a downsampled version of the photo and predicts how to blend a handful of learned 3D color lookup tables into one image-specific correction.
- **`service/`** ships with that paper's own published pretrained weights (`service/weights/model.pt`, sRGB variant, converted into this repo's state-dict layout — see `training/model.py`... credit: original weights from the authors' repo's `pretrained_models/sRGB/`). The LUT-application step is reimplemented with `torch.nn.functional.grid_sample` instead of the original repo's CUDA-only compiled extension, so the identical model runs on plain CPU, Apple Silicon (MPS), or Intel integrated GPU (OpenVINO) with no GPU-specific build step.
- **The 8 style presets** (Natural Light, HDR Punch, Cinematic Teal-Orange, Golden Hour Warm, Moody Matte Film, B&W Dramatic, Clean Commercial, Vibrant Punch) are a separate, deterministic PyTorch-based grading pipeline (`service/presets.py`) applied on top of the model's corrected baseline — not learned, hand-tuned.
- **`training/`** is a from-scratch training pipeline (Mac/MPS) against the MIT-Adobe FiveK dataset, for anyone who wants to train their own model instead of using the shipped pretrained weights. Not needed to run the service.

## Running the service

```
cd service
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # fill in IMMICH_URL / IMMICH_API_KEY if you want Immich import
uvicorn main:app --host 0.0.0.0 --port 5054
```

Or via Docker — build context must be the repo root (not `service/`), since the Dockerfile pulls in `shared/model.py`:

```
docker build -f service/Dockerfile -t photo-enhance .
docker run -p 5054:5054 --env-file service/.env photo-enhance
```

Set `INFERENCE_DEVICE=openvino_gpu` (with `/dev/dri` passed through) to route inference through an Intel integrated GPU instead of CPU — see `service/model_runtime.py`.

## Training your own model

```
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 download_raw.py --n_images 1500   # pulls from MIT's authoritative FiveK source
python3 process_raw.py                     # builds the 480p training set
python3 train.py --batch_size 32 --n_epochs 1000
```

Copy the resulting `checkpoints/best.pt`'s `model` key into `service/weights/model.pt` to deploy it.

## License

Apache-2.0 (see `LICENSE`), matching the license of the original Image-Adaptive-3DLUT repository this project's model architecture and shipped pretrained weights are built on.
