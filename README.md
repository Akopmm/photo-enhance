# photo-enhance

A self-hosted, Lightroom-style photo enhancement service. Point it at a photo — Canon CR3, Sony ARW, or plain JPEG — and it predicts a colour/exposure correction with a small neural network, then renders **18 style variants** on top of it. In *enhanced* mode it also segments the photo — subject, sky, foliage and depth — so those can be graded separately, and suggests compositional crops.

Runs entirely on a home server's CPU. No GPU required, nothing leaves the machine.

Built as a personal alternative to an Adobe Lightroom subscription.

---

## How it works

### The model

A small CNN (~600K parameters) based on [*Learning Image-Adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time*](https://github.com/HuiZeng/Image-Adaptive-3DLUT) (Zeng et al., Apache-2.0). It looks at a downsampled copy of the photo and predicts how to blend a handful of learned 3D colour lookup tables into one image-specific correction.

`service/weights/model.pt` ships **that paper's own published pretrained weights** (sRGB variant, converted into this repo's state-dict layout; originals from the authors' `pretrained_models/sRGB/`). The LUT application is reimplemented with `torch.nn.functional.grid_sample` instead of the original repo's CUDA-only compiled extension, so the identical model runs on plain CPU, Apple Silicon (MPS) or an Intel iGPU (OpenVINO) with no GPU-specific build step.

> If you reimplement this: the LUT axis order is **B, G, R** across the tensor's first three spatial axes (R varies along the *last*). Getting it backwards still trains fine — it's self-consistent — but the published weights then produce garbage. Verified against the original repo's own `IdentityLUT33.txt`.

### Styles (both modes)

18 deterministic looks in `service/presets.py` and `service/cinematic.py`, applied on top of the model's corrected baseline. Hand-tuned parameters, not learned:

**Core (12)** — Natural Light+, HDR Punch, Cinematic Teal-Orange, Golden Hour Warm, Moody Matte Film, B&W Dramatic, Clean Commercial, Vibrant Punch, Bright & Airy, Faded Retro, Deep Contrast Noir, Cool Arctic.

**Cinematic pack (6)** — Teal & Orange, Bleach Bypass, Midnight Blue, Warm Film, Cold Thriller, Golden Epic. What reads as "cinematic" is a small set of ingredients: a filmic curve (lifted blacks, rolled highlights), split toning, restrained saturation, halation and grain. No model involved; toggleable in settings.

### Region-aware grading (enhanced mode)

The LUT model applies **one** colour mapping to every pixel, so it structurally cannot treat a subject differently from its background. Segmentation supplies the *where*:

| Mask | Model | Notes |
|---|---|---|
| Subject | [BiRefNet-lite](https://github.com/zhengpeng7/birefnet), 44M params, MIT | Crisp cutouts; the "Select Subject" equivalent |
| Sky / scene | SegFormer-B0 / ADE20K | Coarse but fine for sky, which is graded softly anyway. Also reports what else is in the frame, which gates the other recipes |
| Depth | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), 24.8M params, Apache-2.0 | Lightroom's Depth Range Mask. Grades by distance, so it still works where subject segmentation finds nothing |

Recipes: **Selective Colour** (subject in colour, background mono), **Subject Pop**, **Sky Drama**, **Depth Pop**, **Aerial Depth**, **Foliage**. Only the ones the photo can actually support are offered — Aerial Depth needs outdoor content, Foliage needs greenery, and the depth looks need the scene to span a depth range at all.

**Masks are computed once and reused.** They are produced at 1024 px, persisted next to the import as PNG, and resampled to whatever the output needs. That is not a shortcut: BiRefNet resizes its input to 1024×1024 internally, so segmenting the full 6000 px image yields the same prediction upscaled. Previously the preview and the full-resolution download segmented independently and disagreed (measured 0.2120 vs 0.1873 sky coverage on one photo), so the file you downloaded was not the grade you approved.

Masking failures degrade to the ordinary global styles rather than failing the import. A recipe is skipped when the mask is degenerate — judged on **confidence as well as area**, because area alone cannot tell "found nothing" from "found something small": a bird on a wire covers 0.9 % of the frame and is exactly the photo Selective Colour is for.

**Region strength.** A slider on each import dials the region recipes back toward the ungraded image; the value is part of the render cache key. It attenuates the whole-frame layers too, not just the masks — in Selective Colour the mono conversion *is* a whole-frame layer, so scaling only the subject mask would leave the background fully black & white however far the slider was pulled back. Downloads take `?strength=0.0…1.0`.

### Crop suggestions (enhanced mode)

Built on the same subject mask, so it's arithmetic rather than guesswork: the subject's mass-weighted centroid is placed on the nearest rule-of-thirds intersection, across 1:1 / 4:5 / 3:2 / 16:9 / 2.39:1.

- Prefers keeping the subject whole; will offer a tighter crop that clips it, reporting `subject_coverage`, with intact crops ranked first.
- Filters out no-op crops (≥95% of the frame kept) and destructive ones (<60% of the subject).
- Each suggestion renders its own preview thumbnail, so the framing is visible before downloading. Downloads take `?crop=<key>`.

Deliberately rules, not a learned aesthetics model — that would be another ~100MB and seconds per photo to land in roughly the same place for the common single-subject case. With no clear subject it falls back to a centred crop and says so.

### Preview-first rendering

Decode and model inference run once at full resolution on import — both are cheap at any size, since the weight-predictor CNN downsamples to 256×256 internally and the LUT is a pointwise op. The genuinely expensive part (the box-blur-based preset effects) only runs at full resolution for a style you actually **download**, and is cached afterwards.

Import is a few seconds; the first full-res download of a style takes a few more, then it's instant. The gallery also shows the original, uncorrected photo for comparison.

### Users

Login is stdlib-only — `hashlib.scrypt` for passwords, HMAC-signed session cookies. No auth framework for what is a handful of accounts on a private network.

Each user holds **their own** Immich URL and API key, so imports read from their own library. Galleries are per-user and every read path checks ownership rather than trusting UUIDs to be unguessable. API keys are never sent to any browser — only a masked preview — and submitting an empty key means "leave unchanged".

---

## Modes

| | classic | enhanced |
|---|---|---|
| Colour correction + 18 style presets | ✓ | ✓ |
| Subject / sky / depth region grading | — | ✓ |
| Crop suggestions | — | ✓ |
| Extra RAM while processing | none | ~1.25 GB |
| Time per photo (6-core i5-10500T) | ~5 s | ~17 s |

Switch in **Settings**. Defaults to `classic`.

---

## Running it

### Docker (recommended)

CI publishes `ghcr.io/akopmm/photo-enhance:latest` on every push to `main`, with the segmentation weights **baked in** — no cold-start download, no dependency on Hugging Face being reachable.

```bash
docker run -d --name photo-enhance -p 5054:5054 \
  -e PHOTO_ENHANCE_ADMIN_USER=you \
  -e PHOTO_ENHANCE_ADMIN_PASSWORD='a-long-password' \
  -v /srv/photo-enhance-data:/app/service/data/renders \
  ghcr.io/akopmm/photo-enhance:latest
```

Building locally — the build context must be the **repo root**, not `service/`, because the Dockerfile pulls in `shared/model.py`:

```bash
docker build -f service/Dockerfile -t photo-enhance .
```

> `torchvision` must be resolved from the same CPU index as `torch`. `timm` pulls it in transitively, and the default PyPI wheel is a CUDA build whose compiled ops don't match (`operator torchvision::nms does not exist`). The Dockerfile installs the pair together.

### From source

```bash
cd service
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5054
```

### First run

The instance starts with no users. Either set `PHOTO_ENHANCE_ADMIN_USER` / `PHOTO_ENHANCE_ADMIN_PASSWORD`, or just open the page — the login screen offers a one-time "create admin" form. That form refuses once any user exists.

Then, in **Settings**, add your Immich URL and an API key scoped to `album.read`, `asset.read`, `asset.view`, `asset.download`.

Everything else — mode, masking toggles, concurrency, JPEG quality, users — is editable there and persisted to `data/renders/_config/` **inside the mounted volume**, so it survives image updates.

> Keep that config path inside the volume. An earlier version wrote it to the volume's *parent*, which lives in the container's ephemeral layer, so every update silently wiped all users, their API keys and the session secret — while login still appeared to work, because the admin re-bootstraps from env.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `PHOTO_ENHANCE_ADMIN_USER` / `_PASSWORD` | — | Bootstraps the first admin only |
| `RENDER_STORAGE_DIR` | `service/data/renders` | Renders **and** config |
| `MAX_CONCURRENT_JOBS` | `3` | Whole-pipeline gate, decode included |
| `IDLE_UNLOAD_MINUTES` | `15` | Drops the model when idle |
| `INFERENCE_DEVICE` | `cpu` | or `openvino_cpu` / `openvino_gpu` |

On concurrency: the gate covers the **entire** pipeline. An earlier version only locked the model call while RAW decode ran unbounded on the default thread pool, so ~30 concurrent imports meant ~30 simultaneous full-resolution decodes (~250-400MB each) and pegged the host. Segmentation has its own stricter `Semaphore(1)` on top.

`INFERENCE_DEVICE=openvino_gpu` (with `/dev/dri` passed through) routes inference through an Intel iGPU. Measured on an i5-10500T this was **no faster** than plain CPU for the global-LUT path — each preset is a separate small dispatch, so overhead swamps the gain. Segmentation is a much better fit for that hardware if you want to revisit it.

---

## Training your own model

Not needed to run the service — it ships with working pretrained weights. This exists if you want to train against the MIT-Adobe FiveK dataset yourself (Mac/MPS or CUDA).

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Downloads + processes FiveK in disk-safe batches, deleting raws as it
# goes. The full raw set is ~290GB; the processed 480p pairs are ~600MB.
python3 fetch_dataset.py --target 5000 --batch 600 --min_free_gb 60

python3 train.py --batch_size 32 --n_epochs 700
python3 compare_models.py --checkpoint checkpoints/best.pt --dump_dir /tmp/cmp
```

`fetch_dataset.py` refuses to start a batch below `--min_free_gb`. An unbounded download once took a laptop down to 121MB free, which is why the guard is there.

To deploy a checkpoint, copy `checkpoints/best.pt`'s `model` key into `service/weights/model.pt`.

Two things worth knowing if you train:
- **PSNR measures agreement with Expert C, not "looks good."** Clamp to [0,1] before scoring — the model legitimately overshoots, and unclamped values inflate MSE for pixels that were never visible.
- `compare_models.py` documents its own biases in the file. The pretrained model was trained on all of FiveK, which includes whatever ends up in your test split, so that comparison favours it.

---

## License

Apache-2.0 (see `LICENSE`), matching the original Image-Adaptive-3DLUT repository this project's architecture and shipped weights come from.
