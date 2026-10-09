"""Phase 1, step 1: block features of every model instance on one dataset (ds003478 by default).

An instance is one untrained model (one weight seed) or one pretrained checkpoint. Each
instance embeds every 2 s window; the windows of each 1-minute block are averaged, giving
one feature vector per block. Probing (phase1_probe.py) and ensembles (phase 2) read these.

Extractors (each with the input its weights expect):
  cnn1d, shallow, eegnet   untrained, 60 channels, 1-40 Hz, 128 Hz ("standard" input)
  eegnet_3ch               untrained EEGNet on C3, Cz, C4: the same input as eegnetv4
  eegnetv4                 pretrained EEGNet, 10 motor-imagery checkpoints (C3, Cz, C4)
  shallow_bcic2a           untrained ShallowFBCSPNet on the same input as shallow_pretrained
  shallow_pretrained       pretrained ShallowFBCSPNet, 2 checkpoints (22 channels, 250 Hz,
                           4-38 Hz, exponential moving standardization: "bcic2a" input)
  scattering               Mallat's wavelet scattering (fixed wavelets, no weights), one
                           instance per largest scale J = 3..8 ("standard" input)

Extractors whose input the dataset lacks (channels, or a preprocessing variant) are skipped.
Writes results/phase1/<dataset>/features/<extractor>/<instance>.npz; existing files are
skipped, so an interrupted job can be resubmitted. Data and weights are downloaded if missing
(weights: bash scripts/00_download_pretrained.sh).

    python scripts/phase1_features.py                    # everything, 30 seeds, ds003478
    python scripts/phase1_features.py --config configs/ds004504.yaml
    python scripts/phase1_features.py --extractors eegnetv4 --seeds 0-4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset, variant_config
from untrained_eeg.models import WaveletScattering, build_model, embed
from untrained_eeg.pretrained import EEGNETV4_CHANNELS, EEGNETV4_DATASETS, SHALLOW_CHECKPOINTS, load_eegnetv4, load_shallow

# Cho2017 and Ofner2017 were trained on another input scale (docs/pretrained_models.md).
EEGNETV4_USED = [d for d in EEGNETV4_DATASETS if d not in ("Cho2017", "Ofner2017")]

EXTRACTORS = {  # name: (group, preprocessing variant, channel subset or None)
    "cnn1d": ("untrained", "standard", None),
    "shallow": ("untrained", "standard", None),
    "eegnet": ("untrained", "standard", None),
    "eegnet_3ch": ("untrained", "standard", EEGNETV4_CHANNELS),
    "eegnetv4": ("pretrained", "standard", EEGNETV4_CHANNELS),
    "shallow_bcic2a": ("untrained", "bcic2a", None),
    "shallow_pretrained": ("pretrained", "bcic2a", None),
    "scattering": ("wavelet", "standard", None),
}
SCATTERING_J = range(3, 9)  # largest wavelet scale 2**J samples: 62 ms to 2 s at 128 Hz


def instances(name, seeds):
    """(instance id, function n_chans, n_times -> model) for every instance of an extractor."""
    if name == "eegnetv4":
        return [(d, lambda c, t, d=d: load_eegnetv4(d, t)) for d in EEGNETV4_USED]
    if name == "shallow_pretrained":
        return [(k, lambda c, t, k=k: load_shallow(k, t)) for k in SHALLOW_CHECKPOINTS]
    if name == "scattering":
        return [(f"J{j}", lambda c, t, j=j: WaveletScattering(t, j)) for j in SCATTERING_J]
    arch = {"eegnet_3ch": "eegnet", "shallow_bcic2a": "shallow"}.get(name, name)
    return [(f"seed{s:03d}", lambda c, t, s=s: build_model(arch, c, t, s)) for s in seeds]


def block_features(model, windows, per_block):
    """Mean feature vector of each block: (n_blocks, n_features), plus block conditions."""
    X, cond = windows
    f = embed(model, X)
    n = len(X) // per_block
    return f[: n * per_block].reshape(n, per_block, -1).mean(axis=1), cond[::per_block][:n]


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--extractors", nargs="+", choices=list(EXTRACTORS), default=list(EXTRACTORS))
parser.add_argument("--seeds", type=seed_range, default=list(range(30)), help="untrained seeds, e.g. 0-29")
args = parser.parse_args()

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
cfg = load_config(args.config) if args.config else load_config()
out_root = Path(cfg["results_dir"]) / "phase1" / cfg["openneuro"]["id"] / "features"
per_block = int(cfg["segment_s"] / cfg["window_s"])  # 30 windows of 2 s per 1-minute block
data = {}  # preprocessing variant -> (participants, {participant_id: (X, cond)})

for name in args.extractors:
    group, variant, subset = EXTRACTORS[name]
    have = {c.upper() for c in cfg["channels"]}
    if (variant != "standard" and variant not in cfg.get("preprocessing", {})) or \
            (subset and not {c.upper() for c in subset} <= have):
        print(f"{name}: skipped, this dataset lacks its input", flush=True)
        continue
    if variant not in data:
        data[variant] = load_dataset(variant_config(cfg, variant))
    participants, windows = data[variant]
    if subset:  # channel indices in the cached order (config names are upper case)
        names = [c.upper() for c in cfg["channels"]]
        idx = [names.index(c.upper()) for c in subset]
        windows = {p: (X[:, idx], cond) for p, (X, cond) in windows.items()}
    pids = participants["participant_id"].tolist()
    _, n_chans, n_times = windows[pids[0]][0].shape
    for inst, make in instances(name, args.seeds):
        out = out_root / name / f"{inst}.npz"
        if out.exists():
            continue
        model = make(n_chans, n_times)
        rows = [block_features(model, windows[p], per_block) for p in pids]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, feats=np.concatenate([f for f, _ in rows]).astype(np.float32),
                 cond=np.concatenate([c for _, c in rows]),
                 participant_id=np.concatenate([[p] * len(c) for p, (_, c) in zip(pids, rows)]),
                 group=group, variant=variant)
        print(f"{name:18s} {inst:18s} {rows[0][0].shape[1]:5d} features", flush=True)
