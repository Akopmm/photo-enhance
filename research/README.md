# research

Experiments that changed a decision, or that failed to.

Each folder holds the scripts that produced the numbers and a `FINDINGS.md`
that states what was asked, what was measured, and what was concluded --
including the conclusions that turned out to be wrong, which are usually the
part worth keeping.

The scripts are meant to stay runnable. They import from `../../service`, so
run them from the repository root with the service's own interpreter:

    PHOTO=/path/to/shot.cr3 ./service/.venv/bin/python \
        research/denoise-speed/validate.py

`validate.py` is the one to reach for: it renders through the function the
download button calls and measures the JPEG that comes out. The others measure
components, which is how four findings in that folder came to be confidently
wrong.

They need a RAW file and, for the denoiser, its weights. Neither is committed:
weights come down with the Docker build, and a test photo is yours to point at.

| folder | question | outcome |
|---|---|---|
| `denoise-speed` | Can renders reach seconds using cheaper denoising? | Yes — 14× at matched noise and detail. Four wrong answers on the way, all measurements of the wrong thing |
