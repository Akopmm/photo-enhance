# Can a render finish in seconds?

**Asked:** could a different model — possibly not a neural network — bring
render time down to seconds?

**Answer:** the cheap methods tested do not replace SCUNet, and the one idea
that looked like a 46× win turned out to be worth half a second. What follows
is the measurement, including two wrong turns, because both were only caught
by checking rather than by reasoning.

## Where the time goes

Full-resolution render of a 24 MP CR3, measured on the optiplex iGPU:

| stage | time |
|---|---|
| LibRaw decode | 2.51 s |
| resize | 0.04 s |
| look (colour maths) | 0.61 s |
| **denoise** | **896.8 s** — 126 tiles × 7.12 s |

Denoise is 99.7% of it. Nothing else is worth optimising.

The cause is architectural: SCUNet has no internal downsampling, so its cost
scales linearly with pixel count. Most denoisers downsample internally, which
is exactly what makes them cheap.

## Wrong turn 1: the test photos had no noise

The first benchmark ran on local RAWs measuring sigma 0.73–2.99. Every one is
below the service's own threshold — `_should_denoise` returns **False** for all
of them. It was measuring how faithfully each method reproduced a denoiser on
photos the service would never denoise.

Timings survived. Every quality number was meaningless. Re-run on a frame at
sigma 3.24 where `_should_denoise` is True.

## Wrong turn 2: the reference was the wrong path

The corrected benchmark compared against "decode 26 MP, denoise 26 MP" and
found half-size decoding 46× faster with no visible difference. That reference
is only what the service does for `size=original`.

Every other preset **already resizes right after the colour model**, so a
Medium render denoises a 3200 px frame: 35 tiles, not 140. Measured both ways,
a Medium render produces **35 tiles either way** — half-size decoding saves the
decode, 0.53 s, and nothing else.

A 46× speedup that applies to one non-default preset, presented as if it
applied to the common path.

## What the numbers say, at the size that matters

Medium preset, 3200 px working size, 26 MP source at sigma 3.24:

| variant | time | speedup | vs reference | denoising strength |
|---|---|---|---|---|
| SCUNet at 3200 (ships today) | 16.03 s | 1× | — | 29.52 dB |
| residual ½ | 5.14 s | 3.1× | 31.23 dB | 37.24 dB |
| residual ¼ | 1.74 s | 9.2× | 30.05 dB | 41.58 dB |
| guided chroma | 0.09 s | 176× | 35.08 dB | 30.56 dB |

"Denoising strength" is PSNR against the untouched frame: **lower means it
changed the photo more**. It is there because PSNR-against-reference cannot
distinguish "matched the denoiser" from "did nothing" — doing nothing scores
29.52 dB purely because the reference does not move the image far either.

Read alone, that table says guided chroma wins: 176× faster, and it changes
the image about as much as SCUNet does.

## What the pictures say

Run `crops2.py` to generate `compare_medium.png` — 100% crops at the Medium
working size, patches chosen automatically as the flattest, highest-frequency
regions rather than by eye. The sheets are not committed: they are crops of
whatever photo you point `PHOTO` at, and this repository is public.

SCUNet is visibly cleaner. Guided chroma and residual ½ both leave grain that
SCUNet removes. The metric could not see this: guided chroma removes **chroma**
noise, which is most of the pixel-difference energy, and leaves **luma** grain,
which is most of what the eye objects to.

`crops.py` shows the same at full sensor resolution, where the gap is wider
still.

## Why every "denoise smaller" idea failed

Three variants were tried that all shrink the image before the network:
residual ½, residual ¼, and a hybrid giving chroma to the guided filter and
luma to SCUNet at reduced resolution. The hybrid is the interesting one,
because it hands the network only the job the filter cannot do.

| variant | time | vs reference | strength |
|---|---|---|---|
| SCUNet at 3200 | 16.03 s | — | 29.52 dB |
| hybrid: filter + luma net ½ | 6.41 s | 35.99 dB | 30.50 dB |
| hybrid: filter + luma net ¼ | 1.83 s | 35.30 dB | 30.53 dB |
| guided chroma alone | 0.09 s | 35.08 dB | 30.56 dB |

All three score the same as the filter **alone**, and the crops confirm it:
adding the network changed nothing a viewer could see. Running SCUNet on luma
at half resolution bought 6.3 seconds of nothing.

The reason is structural, and it explains every earlier residual result too:

> **Grain is high-frequency. Downsampling destroys it before the network sees
> it.** The shrunken image has no grain left to remove, so the network's
> output barely differs from its input, the residual is close to zero, and
> upsampling that residual adds nothing back.

So the family of ideas is dead, not merely untuned. Any scheme that reduces
resolution before the network cannot remove grain, because the grain lives
precisely in the resolution being discarded. The network has to see
full-resolution pixels to fix full-resolution noise, and its cost is therefore
inherent in the pixel count.

That leaves three real levers, and only one of them is unexplored:

1. **Denoise at the delivered size.** Already done — the presets resize before
   denoising, which is why a Medium render is 35 tiles and not 140.
2. **Accept the grain.** That is `denoise_method: fast`, the guided chroma
   filter, at 176x.
3. **A cheaper network per pixel** — untested here. SCUNet is 17.9M parameters
   with no internal downsampling, which is exactly what makes it expensive.
   FFDNet (~0.5M, operates on a pixel-shuffled sub-image) and DRUNet (a U-Net
   that does downsample) are non-blind, taking a noise level the service
   already estimates. This is the remaining avenue.

## The lever that worked: a cheaper network

Resolution reduction is dead, so the remaining option was a network costing
less per pixel while still seeing every pixel. Two candidates, both non-blind
-- they take a noise level, which this service already estimates.

Measured at the Medium working size (3200px), sigma estimated at 4.6:

| variant | time | speedup | vs SCUNet | strength |
|---|---|---|---|---|
| SCUNet 17.9M (ships today) | 16.03 s | 1x | — | 29.52 dB |
| FFDNet 0.85M, sigma 4.6 | 0.50 s | 32x | 32.44 dB | 35.84 dB |
| **FFDNet 0.85M, sigma 9.1** | **0.50 s** | **32x** | **43.25 dB** | **30.14 dB** |
| DRUNet 32.6M, sigma 4.6 | 4.99 s | 3.2x | 31.82 dB | 39.48 dB |
| DRUNet 32.6M, sigma 9.1 | 4.96 s | 3.2x | 38.96 dB | 31.19 dB |

FFDNet at twice the estimated sigma is 43.25 dB from SCUNet -- far closer than
anything else tried, the cheap filter included at 35.08 dB -- and denoises to
almost the same strength. **The crops agree**, which is the first time in this
investigation that the numbers and the pictures pointed the same way: smooth
like SCUNet, highlight detail intact, where every earlier candidate stayed
visibly grainy.

It works for the reason the others failed. FFDNet operates on a 2x2
pixel-shuffled sub-image: a quarter of the spatial positions, but every
original pixel still reaches the network. The grain is never discarded, so
there is something left to remove.

At full resolution on optiplex, 26MP:

| | per tile | 26MP total | peak RSS |
|---|---|---|---|
| SCUNet | 3.40 s | **476 s** | ~2.8 GB compiled |
| FFDNet | 0.30 s | **41.6 s** | 0.59 GB |

Whole-frame in one pass works on Apple silicon (1.89 s, 1.1 GB) but was
OOM-killed twice on optiplex, which has ~7 GB free. It needs the same tiling
SCUNet uses; at 0.30 s a tile that costs nothing.

Three things this measurement does NOT establish:

- **The sigma multiplier is hand-picked.** 2x the estimator worked on one
  photo; at 1x the result is visibly under-denoised. It needs calibrating over
  a set, not a sample.
- **The checkpoint is AWGN-trained**, where the SCUNet in service is
  `color_real`. It held up on real sensor noise here. One frame is not a
  distribution.
- Nothing was compared at `size=original`, only at the Medium working size and
  by tile cost.

## Conclusion

- No cheap substitute for SCUNet was found. Guided chroma is a real option for
  a *fast* mode — 176× faster, chroma noise gone, grain kept — but it is not
  the same output.
- The preset path is already close to optimal: it denoises at the delivered
  size, which is the single most valuable thing it could do.
- `size=original` remains expensive (~9 min for 26 MP) and is inherent: full
  resolution and cheap denoising are the same trade-off in different words.

## What actually made the cheap network match the expensive one

Measured through `validate.py`, on the download, on three photos, on noise AND
detail. Times are a 3200px render on the optiplex CPU.

| photo | quality (SCUNet) | balanced, global level | balanced, shadow level |
|---|---|---|---|
| dark indoor | 207s  2.72 / 1.53 | 14.2s  11.33 / 12.35 | **14.2s  2.40 / 1.50** |
| daylight | 201s  1.76 / 11.13 | 14.3s  1.85 / 9.98 | **15.0s  2.07 / 11.27** |
| outdoor | 210s  7.93 / 7.90 | 14.5s  8.11 / 7.76 | **14.8s  8.02 / 7.45** |

*(noise in the shadow band / detail; lower noise and higher detail are better)*

FFDNet is non-blind, so everything depends on the level it is told, and a
multiple of the global estimate cannot be that level. Two of these photos
measured 8.6 and 10.2 globally and needed opposite treatment: one carried 24.4
in its shadows, the other 10.2. A multiplier tuned on the first destroys the
second (detail 11.13 -> 6.51) and one tuned on the second leaves the first
untouched (shadows 11.33 against SCUNet's 2.72).

The global figure is a floor taken from the flattest blocks, and in a dark
frame those are the calm bright areas. So the level now comes from the shadow
band itself -- where sensor noise is worst, because it is signal-dependent and
the read floor dominates where there is least signal, and where the eye finds
it. Never below the global figure, so a frame with no shadows is no worse off.

That single change makes a 0.85M-parameter network match a 17.9M one on both
axes at **14x the speed**. It is the same reason the whole "denoise smaller"
family failed: what matters is not how much work the network does but whether
it is looking at the thing that needs fixing.

## Every wrong answer here failed the same way

Four findings in this investigation were reported confidently and were wrong.
Each was a real measurement of the wrong thing.

| claimed | measured | what the download actually did |
|---|---|---|
| half-size decode is 46x faster | against `size=original` | presets already resize; worth 0.53s |
| the estimator is fine | greyscale average | channel noise cancels; 3.37 read as 1.91 |
| the gate is fine once the estimator is | the 1280px preview | downscaling averages noise; 3.37 read as 2.47 |
| balanced matches quality | the decoded image, sigma 9.5 | denoise runs AFTER the colour model, at sigma 24 |

The pattern is not carelessness about any one number. It is that a component
can be measured correctly and still say nothing about the product, and there
was no habit of checking the product. `validate.py` exists to be that habit:
it renders through `render_full_style`, the function the download button
calls, and measures the JPEG that comes out.

Two things it measures that a single figure hides:

* **Noise per brightness band.** A dark indoor frame measured 0.93 overall and
  carried 11.33 in its shadows. Overall figures average the shadows, where
  noise lives, into the bright areas that dominate the block count.
* **Detail as well as noise.** Both are high-frequency, so "less noise" and
  "less detail" are the same measurement unless they are taken separately --
  noise in the flattest blocks, detail in the busiest. Every scale above 2.5
  beat SCUNet on noise while scoring below it on detail. That is smearing, and
  a single number calls it an improvement.

## Method notes

`nets.py` needs the KAIR definitions and weights, neither committed:

    pip download / curl the two networks and weights into a directory, then
    WEIGHTS=/that/dir PYTHONPATH=/that/dir PHOTO=/path/shot.cr3 \
      ./service/.venv/bin/python research/denoise-speed/nets.py

  * network_ffdnet.py, network_unet.py, basicblock.py from cszn/KAIR (MIT)
  * ffdnet_color.pth, drunet_color.pth from its v1.0 release


- Reference cached to `ref.npy`; delete to rebuild.
- Crop patches are chosen by high-frequency energy in flat regions, with blown
  highlights and pure black excluded, so the comparison cannot be flattered by
  choosing kind patches.
- `bench.py` targets `size=original`; `bench2.py` targets the preset path.
  Read `bench2.py` first — it answers the question that matters.
