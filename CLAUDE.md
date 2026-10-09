# CLAUDE.md

Context for AI assistants (and humans) working in this repo.

## Project
Brainhack Montreal Fall 2026, prototype, aiming at a short paper. Question: how do
untrained (random-weight) CNNs compare with trained (pretrained) models on EEG, across
several tasks on open datasets? Variability (across seeds, and numerical via Fuzzy PyTorch / MCA)
is part of the comparison. Lead: Roy Seo (@royjoysoy).
The repo is public and used by Brainhack beginners: keep it clean, short, and well commented.

## Design decisions (settled)
- Untrained models: small custom 1D CNN, ShallowFBCSPNet, EEGNet. Random weights, eval mode, never trained.
  The two braindecode models are copied as plain PyTorch in `models.py` (no braindecode/MNE in the
  Fuzzy environment); `tests/test_models.py` checks they stay bit-identical to braindecode 1.8.1.
- Trained models: all publicly available pretrained checkpoints for scalp EEG (CNNs and foundation
  models such as LaBraM). We do not train our own models. Pretrained weights must load into
  plain-PyTorch copies in `models.py` (no braindecode in Fuzzy).
- Open datasets only (no registration-gated data). An evaluation dataset must not be in a
  checkpoint's pretraining data (leakage).
- Same seed ≠ same weights across PyTorch builds (last-bit differences, e.g. torch 2.14 vs Fuzzy 2.2):
  run the seed baseline with the same build as the Fuzzy runs (normal rounding), or reuse saved weights.
- Untrained and pretrained networks are compared the same way: frozen feature extractor (head removed)
  + downstream probe. Variability is measured on the probe, not on embeddings.
- Probe on ds003478: logistic regression, high vs low BDI, stratified 5-fold CV with a fixed split.
- Data (first task): ds003478 v1.1.0, run-01 only, 1-minute eyes-open/closed blocks → 2 s windows, participant 544 excluded.
- BDI cutoffs: ≤6 low / ≥17 high (equivalent to the original study's <7 / >16 for integer scores). Confirmed.
- License MIT.

## Open items (ask Roy, don't assume)
- Pretrained checkpoints: first inventory in `docs/pretrained_models.md`; "?" cells and its open decisions remain.
- Which extra tasks/datasets (candidates in `docs/datasets.md`), and their probes (trial-level, multi-class).
- Fuzzy PyTorch image name/tag: placeholders marked `TODO(fuzzy-image)` in `environment/` and `slurm/`.
- Shared data path in Rorqual `/project` space.

## Data quirks (ds003478)
- Trigger codes 1–6 repeat every 500 ms within a 1-minute block; odd = eyes closed, even = eyes open, one code per block. Codes 11–16 are 2 s ticks (ignored), 17 = start/finish.
- Some blocks are incomplete (recording started late); blocks < 60 s are skipped.
- Some files have an extra EKG channel → we pick an explicit list of 60 scalp channels (config `channels`).
- 516 has no run-02 (irrelevant, we use run-01). 544: invalid participant, excluded.
- HEOG/VEOG may be swapped; some channels were interpolated by the authors; no raw data to revert to.
- Run-02 was never quality-checked by the dataset authors.
- After exclusions and cutoffs: 113 participants, 74 low, 39 high.

## Conventions
- Data path from `UNTRAINED_EEG_DATA` env var (or `data_dir` in config). Never hardcode it.
- Never commit `data/` or `results/`.
- Preprocessing runs outside the Fuzzy container; scripts 02–04 read only the cache (`build=False`).
- One process = one MCA sample (`03_mca_variability.py --sample i`).
- Outside the container, MCA samples must be bit-identical (sanity check).
- Scripts put `src/` on `sys.path` themselves; there is no package install step.
- Commits must use an email verified on the `royjoysoy` GitHub account.
