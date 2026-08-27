# photo-enhance

A self-hosted, Lightroom-style photo enhancement service. Point it at a photo — Canon CR3, Sony ARW, or plain JPEG — and it predicts a colour/exposure correction with a small neural network, then renders **18 style variants** on top of it. In *enhanced* mode it also segments the photo — subject, sky, foliage and depth — so those can be graded separately, and suggests compositional crops.

Runs on a home server. Nothing leaves the machine, and every model is Apache-2.0 or MIT.

Built as a personal alternative to an Adobe Lightroom subscription.

![The full path of a RAW file through photo-enhance: five neural networks and one look engine,
showing which run on the CPU and which on the integrated GPU](docs/pipeline.jpg)

*Every panel above is a real intermediate from one Canon CR3 — including the three masks, which are
what BiRefNet, UPerNet and Depth Anything actually produced on that frame. Timings measured on the
deploy box, a six-core i5-10500T with UHD 630 graphics.*

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
| Sky / scene | [UPerNet / ConvNeXt-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny), 60.2M params, MIT | ADE20K's 150 classes: the sky mask, the foliage union (tree ∪ grass ∪ plant), and the scene inventory that decides which looks are offered at all |
| Depth | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), 24.8M params, Apache-2.0 | Lightroom's Depth Range Mask. Grades by distance, so it still works where subject segmentation finds nothing |

**Every model is permissive** (Apache-2.0 or MIT) as of 2026-08-26. SegFormer-B0 was replaced by
UPerNet/ConvNeXt-tiny for that reason — the NVIDIA weights permit *"research or evaluation purposes
only"*. Checked before swapping, on the masks the service actually consumes: foliage IoU 0.82/0.92
across two photos, and while sky scored IoU 0.31, by eye UPerNet is the better of the two (SegFormer's
sky came out patchy and leaked into blurred foreground). Costs 60M params against 3.8M, once per import.

### Denoise

[SCUNet](https://github.com/cszn/SCUNet) (Apache-2.0), the authors' **real-noise** model rather than a
synthetic-Gaussian one — which is why it holds up on an actual high-ISO CR3. Measured on one: sigma
4.0 → 0.8 across the frame and 8.1 → 0.3 at 1:1, with eyelashes and catchlights intact.

`denoise_mode: auto` (default) measures the corrected baseline and only denoises above sigma 3, so
clean frames are neither softened nor charged for it. This is the most expensive model in the service
— unlike the others it has no internal downsampling — and a noisy import costs ~87 s on the deploy box
against ~15 s for a clean one.

**The amount is a live slider** because both baselines are stored, untouched and fully denoised: any
amount is a blend between them, so it responds in ~46 ms with no model run. Full-resolution renders
tile the model with feathered overlap (measured 0.14/255 mean difference from a single pass) and blend
to the same amount — a denoised preview with a noisy download would be the same divergence bug the
masking rework removed.

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

### The editor

Two screens: a **Library** of your photos, and an **Editor** you open one into. On a wide screen the
photo sits left and every control right, so you never scroll between the image and the thing that
changes it.

- **Looks** are a filmstrip grouped into *For this photo* (the region-aware ones, which depend on
  what the segmenters found), *Signature* and *Cinematic* — rather than one wall of 24 thumbnails.
- **Strength** and **Denoise** are sliders that re-render the large preview as you drag them. Both
  are arithmetic over things already computed, so they respond in tens of milliseconds with no model
  run: strength blends toward the ungraded baseline, and denoise blends between the two stored
  baselines (untouched and fully denoised).
- **Crop** is drawn *on* the photo — dimmed surround, thirds guides, corner handles. Drag inside to
  reposition, drag a corner to resize; that detaches from the preset and sends an explicit rectangle.
- Press and hold the image to compare against the original.

### Preview-first rendering

Decode and model inference run once at full resolution on import — both are cheap at any size, since
the weight-predictor CNN downsamples to 256×256 internally and the LUT is a pointwise op. Masks are
computed once at 1024 px and persisted; the baseline is stored at 1280 px so the editor's large
preview can be re-rendered from it at any look and strength and still be sharp.

Full-resolution renders happen only for what you actually **download**, run as a job with real
progress (`fetching → decoding → colour model → removing noise → applying the look → cropping →
saving`), and are cached by every parameter that affects the result. Deleting a photo cancels any
render still working on it.

### Output size and crops

Downloads can be **Original**, or ~5 / ~3 / ~1 MB. The resize happens straight after the colour
model, so denoise, styling and cropping all run smaller — on a 26 MP frame that is 140 denoise tiles
against 48. Sizes are labelled in MB and the encoder steps quality down until it fits, because a
denoised frame compresses far better than a noisy one.

Cropping likewise happens **before** the expensive stages rather than on the finished JPEG, which
also removes a second round of JPEG compression on every cropped download.

Crop presets lead with the ratios Instagram actually accepts — post outside this range and it
re-crops for you: **Story/Reel 9:16**, **Portrait 4:5** (the tallest allowed), **Square 1:1**,
**Landscape 1.91:1** (the widest allowed) — followed by 3:2, 16:9 and 2.39:1.

### Users

Login is stdlib-only — `hashlib.scrypt` for passwords, HMAC-signed session cookies. No auth framework for what is a handful of accounts on a private network.

Each user holds **their own** Immich URL and API key, so imports read from their own library. Galleries are per-user and every read path checks ownership rather than trusting UUIDs to be unguessable. API keys are never sent to any browser — only a masked preview — and submitting an empty key means "leave unchanged".

---

## Modes

| | classic | enhanced |
|---|---|---|
| Colour correction + 18 style presets | ✓ | ✓ |
| Subject / sky / foliage / depth region grading | — | ✓ |
| Crop suggestions | — | ✓ |
| Denoise (when the photo needs it) | — | ✓ |
| Import time, 26 MP CR3 (6-core i5-10500T) | ~5 s | ~15 s clean, ~87 s if denoised |
| Container peak while importing | ~1 GB | ~2.5 GB, ~4.5 GB if denoised |

Switch in **Settings**. Defaults to `classic`.

---

## Running it

### Docker (recommended)

There is a ready-made [`docker-compose.example.yml`](docker-compose.example.yml) at the repo root — copy it to `docker-compose.yml`, change the password, `docker compose up -d`.

> **Published image is `linux/amd64` only.** On arm64 (Apple silicon, Raspberry Pi, most ARM NAS boxes) build it yourself: `docker build -f service/Dockerfile -t photo-enhance .` from the repo root. The image also bundles an Intel GPU runtime that is dead weight on non-Intel hosts.

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
| `MAX_CONCURRENT_JOBS` | `2` | Whole-pipeline gate, decode included |
| `DENOISE_DEVICE` | `auto` | `auto` / `cpu` / `gpu`. **Set `cpu` on an Intel iGPU host with <16GB RAM** — see below |
| `IDLE_UNLOAD_MINUTES` | `15` | Drops the model when idle |
| `INFERENCE_DEVICE` | `cpu` | or `openvino_cpu` / `openvino_gpu` |

On concurrency: the gate covers the **entire** pipeline. An earlier version only locked the model call while RAW decode ran unbounded on the default thread pool, so ~30 concurrent imports meant ~30 simultaneous full-resolution decodes (~250-400MB each) and pegged the host. Segmentation has its own stricter `Semaphore(1)` on top.

`INFERENCE_DEVICE=openvino_gpu` (with `/dev/dri` passed through) routes the colour model through an Intel iGPU. An earlier note here claimed this was "no faster than CPU" — **that measurement was invalid**: it was taken with no Intel compute runtime installed, so OpenVINO reported no GPU and silently ran on CPU. It was comparing CPU against CPU. Untested since.

The denoiser is the one model where the iGPU clearly pays: **5.67 → 3.39 s** per 512 px tile on a UHD 630. Two things to know before enabling it:

> ⚠️ **`DENOISE_DEVICE=auto` needs ~16GB of RAM on an Intel iGPU host.** Compiling SCUNet for OpenVINO calls `ov.convert_model`, which peaks **above 12GB**, once per process — 12GB and 10GB containers and a 16GB CI runner are all OOM-killed doing it. On a smaller box set `DENOISE_DEVICE=cpu`. Hosts with no `/dev/dri` never reach that code path, so this only affects Intel iGPU users who opt in.

> INT8 quantisation does **not** help here and is not worth trying: built with NNCF and calibrated on real tiles, it measured **20% slower** (4.11 vs 3.41 s/tile), because UHD 630 is Gen9.5 and has no usable INT8 dot-product path. FP16 is already the iGPU default.

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

## Models and licences

| stage | model | params | licence | device |
|---|---|---|---|---|
| Colour | Image-Adaptive-3DLUT | 593 K | Apache-2.0 | CPU |
| Denoise | SCUNet (real-noise) | 17.9 M | Apache-2.0 | iGPU, falls back to CPU |
| Subject | BiRefNet-lite | 44 M | MIT | CPU |
| Scene / sky | UPerNet ConvNeXt-T | 60.2 M | MIT | CPU |
| Depth | Depth Anything V2 Small | 24.8 M | Apache-2.0 | CPU |

All permissive. SegFormer was used for scene segmentation until its NVIDIA licence — *"research or
evaluation purposes only"* — made it unsuitable; UPerNet replaced it and produced a **better** sky
mask in the comparison.

Only the denoiser uses the integrated GPU, because it is the only model with no internal
downsampling and therefore the only one genuinely compute-bound (2.52× on an Intel UHD 630). It costs
~680 MB of Intel driver mapped permanently into the process, so `denoise_device` can be set to `cpu`
if you would rather have the memory. Gen9.5 hardware needs Intel's *legacy* 24.35 driver line; the
Dockerfile pins it.

### If you want to use this commercially

The models are the easy part — all five are Apache-2.0 or MIT. Three other things deserve a look, in
descending order of how much they matter. **None of this is legal advice.**

1. **LibRaw is LGPL-2.1 / CDDL-1.0** (dual, with a paid commercial option). It is what actually
   decodes your CR3 files, reached through `rawpy` — `rawpy` itself is MIT, but the decoder underneath
   is not. Running this **as a service** does not trigger distribution obligations at all. Shipping it
   as a **closed-source binary** does: you would need to keep LibRaw dynamically linked and allow
   relinking, or buy LibRaw's commercial licence.
2. **The shipped 3D-LUT weights were trained on MIT-Adobe FiveK**, whose dataset terms are
   research-oriented. The architecture's Apache-2.0 licence covers the code, not the question of what
   the weights were learned from. Avoidable if it matters: retrain the LUT on permissively licensed
   pairs — `training/` already does exactly this.
3. **BiRefNet's weights card declares no licence**, even though its code repository is MIT. Worth an
   upstream issue asking them to state it explicitly.

Everything else in the runtime is permissive: torch, numpy and Pillow are BSD-family; transformers,
timm, kornia, OpenVINO and python-multipart are Apache-2.0; FastAPI, einops and rawpy are MIT.

## License

Apache-2.0 (see `LICENSE`), matching the original Image-Adaptive-3DLUT repository this project's
architecture and shipped weights come from. Vendored third-party code carries its own notice — see
`service/vendor/NOTICE` for SCUNet.
