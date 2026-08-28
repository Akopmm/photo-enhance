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

## Conclusion

- No cheap substitute for SCUNet was found. Guided chroma is a real option for
  a *fast* mode — 176× faster, chroma noise gone, grain kept — but it is not
  the same output.
- The preset path is already close to optimal: it denoises at the delivered
  size, which is the single most valuable thing it could do.
- `size=original` remains expensive (~9 min for 26 MP) and is inherent: full
  resolution and cheap denoising are the same trade-off in different words.

## Method notes

- Reference cached to `ref.npy`; delete to rebuild.
- Crop patches are chosen by high-frequency energy in flat regions, with blown
  highlights and pure black excluded, so the comparison cannot be flattered by
  choosing kind patches.
- `bench.py` targets `size=original`; `bench2.py` targets the preset path.
  Read `bench2.py` first — it answers the question that matters.
