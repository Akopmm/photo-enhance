# research

Experiments that changed a decision, or that failed to.

Each folder holds the scripts that produced the numbers and a `FINDINGS.md`
that states what was asked, what was measured, and what was concluded --
including the conclusions that turned out to be wrong, which are usually the
part worth keeping.

The scripts are meant to stay runnable. They import from `../../service`, so
run them from the repository root with the service's own interpreter:

    ./service/.venv/bin/python research/denoise-speed/bench.py

They need a RAW file and, for the denoiser, its weights. Neither is committed:
weights come down with the Docker build, and a test photo is yours to point at.

| folder | question | outcome |
|---|---|---|
| `denoise-speed` | Can renders reach seconds using cheaper denoising? | No replacement found; measurement corrected a wrong premise twice |
