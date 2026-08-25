# photo-enhance

A self-hosted Lightroom-style photo enhancement service: point it at a photo (Canon CR3, Sony ARW, or plain JPEG), it predicts a color/exposure correction with a small neural network, then renders 12 distinct style variants (natural, HDR, cinematic teal-orange, B&W, bright & airy, faded retro, and more) on top of that correction. Built to run entirely on a home server's CPU (no GPU required), with a browser UI for importing from an [Immich](https://immich.app) photo library or a direct upload.

## Why

Built as a personal alternative to an Adobe Lightroom subscription.

## How it works

- **The model** is a small CNN (~600K parameters) based on the architecture from [*Learning Image-Adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time*](https://github.com/HuiZeng/Image-Adaptive-3DLUT) (Zeng et al., Apache-2.0). It looks at a downsampled version of the photo and predicts how to blend a handful of learned 3D color lookup tables into one image-specific correction.
- **`service/`** ships with that paper's own published pretrained weights (`service/weights/model.pt`, sRGB variant, converted into this repo's state-dict layout — see `training/model.py`... credit: original weights from the authors' repo's `pretrained_models/sRGB/`). The LUT-application step is reimplemented with `torch.nn.functional.grid_sample` instead of the original repo's CUDA-only compiled extension, so the identical model runs on plain CPU, Apple Silicon (MPS), or Intel integrated GPU (OpenVINO) with no GPU-specific build step.
- **12 style presets** (Natural Light, HDR Punch, Cinematic Teal-Orange, Golden Hour Warm, Moody Matte Film, B&W Dramatic, Clean Commercial, Vibrant Punch, Bright & Airy, Faded Retro, Deep Contrast Noir, Cool Arctic) are a separate, deterministic PyTorch-based grading pipeline (`service/presets.py`) applied on top of the model's corrected baseline — not learned, hand-tuned.
- **Region-aware grading** (enhanced mode): the LUT model applies one colour mapping to every pixel, so it structurally cannot treat a subject differently from its background. Segmentation supplies the "where" — [BiRefNet-lite](https://github.com/zhengpeng7/birefnet) (44M params, MIT) for subject cutouts and SegFormer-B0/ADE20K for sky — enabling *Selective Colour* (subject in colour, background mono), *Subject Pop* and *Sky Drama*. Masking failures degrade to the ordinary global styles rather than failing the import, and a recipe is skipped when mask coverage looks degenerate (<1% or >95%), since grading through such a mask looks like a bug.
- **Composition-aware crop suggestions**: built on the same subject mask, so it's arithmetic rather than guesswork — the subject's mass-weighted centroid is placed on the nearest rule-of-thirds intersection. Prefers keeping the subject whole but will offer a tighter crop that clips it, reporting how much survives; no-op crops (keeping ≥95% of the frame) and destructive ones (<60% of the subject) are filtered out. Each suggestion renders its own preview thumbnail so the framing is visible before downloading, and downloads take `?crop=<key>`.
- **Multi-user with per-user Immich keys**: login is stdlib-only (`hashlib.scrypt` + HMAC-signed cookies — no auth framework for a handful of accounts). Each user holds their own Immich URL and API key so imports read from their own library, galleries are per-user, and every read path checks ownership rather than trusting UUIDs to be unguessable. API keys are never sent to any browser, only a masked preview.
- **Preview-first rendering**: decode + model inference happen once at full resolution on import (both are cheap regardless of resolution — the weight-predictor CNN always downsamples its input to 256×256 internally, and LUT color correction is a pointwise per-pixel operation, so it costs about the same at any output size). The genuinely expensive part — the deterministic style-preset box-blur effects — only runs at full resolution for a style you actually download, and is cached after that (`pipeline.render_full_style`). Import is a few seconds; a full-res download of one style is a few more seconds the first time, instant after. The gallery also shows the original, uncorrected photo alongside the style renders for comparison.
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

CI (`.github/workflows/build.yml`) builds and publishes this image to `ghcr.io/akopmm/photo-enhance:latest` on every push to `main` — deploy hosts can just `docker pull` that instead of building locally.

Set `INFERENCE_DEVICE=openvino_gpu` (with `/dev/dri` passed through) to route inference through an Intel integrated GPU instead of CPU — see `service/model_runtime.py`. Measured on a 6-core i5-10500T this was *no faster* than plain CPU for the global-LUT path, because each style preset is a separate small dispatch; segmentation is a much better fit for that hardware if you want to revisit it.

On first run the instance has no users: `PHOTO_ENHANCE_ADMIN_USER` / `PHOTO_ENHANCE_ADMIN_PASSWORD` bootstrap the first admin, or the login page offers a one-time "create admin" form. Everything else — mode, masking toggles, concurrency, quality, users and their Immich keys — is editable at `/settings` and persisted under the render volume (`data/renders/_config`), so it survives image updates.

### Modes

| | classic | enhanced |
|---|---|---|
| Global colour correction + style presets | yes | yes |
| Subject / sky region grading | no | yes |
| Crop suggestions | no | yes |
| Extra RAM while processing | none | ~1.2 GB |
| Time per photo (i5-10500T) | ~5 s | ~17 s |

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
