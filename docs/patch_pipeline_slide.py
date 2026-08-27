"""Corrects three things baked into docs/pipeline.jpg as pixels.

The slide's generator was not kept, so these are pixel edits rather than a
re-render. Everything except the badge label is done by filling or shifting,
so there is no body font to mismatch. Measured coordinates, not eyeballed:
the card background is a flat #141B2B, the badge occupies x1107-1218 /
y827-855, and the purple params badge ends at x1102.

  1. "iGPU 2.5x" -> "iGPU 1.7x". The 2.52x figure was taken on a busy box;
     re-measured idle it is 5.67 -> 3.39 s per tile.
  2. the three third-party licence lines, removed
  3. "Lightroom's Depth Range Mask" line, removed and the line below pulled up
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATH = "docs/pipeline.jpg"
BG = (20, 27, 43)
PILL = (1107, 827, 1218, 855)
GLYPH, FILL = (97, 173, 165), (21, 63, 60)

im = Image.open(PATH).convert("RGB")
a = np.asarray(im).copy()

for x0, x1 in ((330, 700), (1130, 1500), (1940, 2310)):      # licence lines
    a[1124:1154, x0:x1] = BG
x0, x1 = 1940, 2310                                           # depth card
a[1186:1212, x0:x1] = a[1210:1236, x0:x1].copy()
a[1212:1240, x0:x1] = BG

im = Image.fromarray(a)
d = ImageDraw.Draw(im)

# The pulled-up line started mid-sentence ("... Depth Range Mask -- no subject
# decision needed."), so it now opens lowercase after a full stop. Redraw it.
# The body font is Helvetica 19px: it renders this string at 242px against the
# 244px actually measured in the image, which is how it was identified.
d.rectangle([1947, 1186, 2310, 1212], fill=BG)
body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
d.text((1947, 1188), "No subject decision needed.", font=body, fill=(125, 134, 153))
d.rectangle([1104, 824, 1221, 859], fill=BG)                  # clear old badge
d.rounded_rectangle(list(PILL), radius=7, fill=FILL)
label = "iGPU 1.7x"
size = 16
while size > 8:
    f = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    if d.textlength(label, font=f) <= 88:
        break
    size -= 1
w = d.textlength(label, font=f)
bb = f.getbbox(label)
d.text((PILL[0] + (PILL[2]-PILL[0]-w)/2,
        PILL[1] + (PILL[3]-PILL[1]-(bb[3]-bb[1]))/2 - bb[1]), label, font=f, fill=GLYPH)
im.save(PATH, quality=94, subsampling=0)
print(f"patched {PATH}: badge font {size}px, label {w:.0f}px (original glyph run 87px)")
